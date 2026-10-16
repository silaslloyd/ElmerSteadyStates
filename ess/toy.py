"""
Stand-in model for testing the solvers without Elmer: a conducting ice column with subtemperate
frictional heating at the bed, plus a thickness-like variable. Same interface as ElmerModel.

    T_t = T_zz,  0 < z < 1,      T(1) = Ts  (the parameter p)
    -T_z(0) = q + F0 H exp(T_b / delta)      (geothermal + frictional heat; Tm = 0)
    H_t = r (1 - H)

Steady states: H = 1, T linear, and T_b solves  Ts = T_b - q - F0 exp(T_b / delta)  -- the S-curve's
cold limb and middle branch. Ts(T_b) has a maximum (the fold) at T_b* = delta ln(delta / F0); for
Ts < Ts* the colder state is stable, the warmer one unstable. The finite-difference discretisation
(ghost-node flux condition) reproduces linear profiles exactly, so these are also the discrete
steady states. Time stepping: backward Euler. coupling="staggered" (default) lags the frictional
heating by one step, as Elmer's Stokes -> heat splitting does, so each step is linear and always
solvable; coupling="implicit" solves the full nonlinear step by Newton (singular when sigma dt = 1 on
an unstable branch). Fixed points are the same either way.
"""
from __future__ import annotations

import numpy as np

from .model import MapSpec, Model, ModelFailure
from .statefile import Block, StateLayout


class ToyColumn(Model):
    def __init__(self, N=40, q=0.5, F0=1.0, delta=0.1, r=0.5, T_scale=1.0, H_scale=1.0, noise=0.0, seed=0,
                 coupling="staggered"):
        self.coupling = coupling
        self.N, self.q, self.F0, self.delta, self.r = N, q, F0, delta, r
        self.h = 1.0 / N
        self.z = np.arange(N) * self.h
        self.layout = StateLayout([Block("T", np.arange(1, N + 1, dtype=np.int64), T_scale),
                                   Block("H", np.array([1], np.int64), H_scale)])
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.nrun = 0

    # ---- continuous right-hand side and its Jacobian
    def rhs(self, x, Ts):
        N, h = self.N, self.h
        T, H = x[:N], x[N]
        g = self.q + self.F0 * H * np.exp(T[0] / self.delta)
        f = np.empty(N + 1)
        f[0] = (2 * T[1] - 2 * T[0] + 2 * h * g) / h ** 2
        f[1:N - 1] = (T[2:] - 2 * T[1:N - 1] + T[:N - 2]) / h ** 2
        f[N - 1] = (Ts - 2 * T[N - 1] + T[N - 2]) / h ** 2
        f[N] = self.r * (1 - H)
        return f

    def jac(self, x, Ts):
        N, h = self.N, self.h
        T, H = x[:N], x[N]
        e = self.F0 * np.exp(T[0] / self.delta)
        J = np.zeros((N + 1, N + 1))
        i = np.arange(1, N - 1)
        J[i, i - 1] = J[i, i + 1] = 1 / h ** 2
        J[i, i] = -2 / h ** 2
        J[0, 0] = (-2 + 2 * h * H * e / self.delta) / h ** 2
        J[0, 1] = 2 / h ** 2
        J[0, N] = 2 * h * e / h ** 2
        J[N - 1, N - 1] = -2 / h ** 2
        J[N - 1, N - 2] = 1 / h ** 2
        J[N, N] = -self.r
        return J

    def _split(self, x, Ts):
        """rhs = L x + c(Ts) + n(x): linear part L, constant c, nonlinear basal source n (node 0)."""
        N, h = self.N, self.h
        L = self.jac(np.append(np.full(N, -np.inf), 0.0), Ts)   # exp(-inf) = 0: linear part only
        L[0, N] = 0.0
        c = np.zeros(N + 1)
        c[0] = 2 * self.q / h
        c[N - 1] = Ts / h ** 2
        c[N] = self.r
        nx = np.zeros(N + 1)
        nx[0] = 2 * self.F0 * x[N] * np.exp(x[0] / self.delta) / h
        dn = np.zeros((N + 1, N + 1))
        dn[0, 0] = nx[0] / self.delta
        dn[0, N] = 2 * self.F0 * np.exp(x[0] / self.delta) / h
        return L, c, nx, dn

    def step(self, x, Ts, dt):
        if self.coupling == "staggered":
            with np.errstate(over="ignore", invalid="ignore"):
                L, c, nx, _ = self._split(x, Ts)
            y = np.linalg.solve(np.eye(len(x)) - dt * L, x + dt * (c + nx))
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e3:
                raise ModelFailure("toy: state blew up")
            return y
        y = x.copy()
        for _ in range(50):
            G = y - x - dt * self.rhs(y, Ts)
            d = np.linalg.solve(np.eye(len(y)) - dt * self.jac(y, Ts), -G)
            y += d
            if np.max(np.abs(d)) < 1e-14 * (1 + np.max(np.abs(y))):
                break
        else:
            raise ModelFailure("toy: implicit step did not converge")
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e3:
            raise ModelFailure("toy: state blew up")
        return y

    def run(self, x, p, spec: MapSpec, tag="run", want_diag=False):
        self.nrun += 1
        y = np.asarray(x, float).copy()
        for _ in range(spec.n_steps):
            y = self.step(y, p, spec.dt)
        if self.noise:
            y = y + self.noise * self.rng.standard_normal(len(y))
        diag = {"Tb": (np.array([1]), y[:1].copy())} if want_diag else {}
        return y, diag

    def map_jacobian(self, x, p, spec: MapSpec):
        """Exact D Phi (product of backward-Euler step Jacobians)."""
        M = np.eye(len(x))
        y = np.asarray(x, float).copy()
        for _ in range(spec.n_steps):
            if self.coupling == "staggered":
                L, _, _, dn = self._split(y, p)
                M = np.linalg.solve(np.eye(len(y)) - spec.dt * L, (np.eye(len(y)) + spec.dt * dn) @ M)
                y = self.step(y, p, spec.dt)
            else:
                y = self.step(y, p, spec.dt)
                M = np.linalg.solve(np.eye(len(y)) - spec.dt * self.jac(y, p), M)
        return M

    # ---- exact steady states
    def Ts_of_Tb(self, Tb):
        return Tb - self.q - self.F0 * np.exp(Tb / self.delta)

    def fold(self):
        Tb = self.delta * np.log(self.delta / self.F0)
        return Tb, self.Ts_of_Tb(Tb)

    def steady_state(self, Tb):
        Ts = self.Ts_of_Tb(Tb)
        T = Tb + (Ts - Tb) * self.z
        return np.append(T, 1.0), Ts

    def steady_Tb(self, Ts, branch="cold"):
        """Solve Ts = Tb - q - F0 exp(Tb/delta) for Tb on the cold or the warm (middle) branch."""
        Tbf, Tsf = self.fold()
        if Ts > Tsf:
            raise ValueError("no steady state beyond the fold")
        lo, hi = (Tbf - 50.0, Tbf) if branch == "cold" else (Tbf, Tbf + 5.0)
        f = lambda t: self.Ts_of_Tb(t) - Ts
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (f(lo) < 0) == (f(mid) < 0):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
