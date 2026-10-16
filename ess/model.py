"""
Abstract time-step map and the scaled fixed-point problem the solvers work on.

A Model advances a state x (physical units) by n_steps of size dt at parameter p:
    x_out = Phi(x; p)
Steady states are fixed points, F(y, p) = Phi(y) - y = 0, written in scaled variables y = x / s
(s = per-block scale from the config) so that all entries are O(1) and norms mix blocks sensibly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .statefile import StateLayout

log = logging.getLogger("ess")


class ModelFailure(RuntimeError):
    """The model could not be evaluated (solver crash, non-convergence, ...)."""


@dataclass(frozen=True)
class MapSpec:
    dt: float
    n_steps: int
    coupled_iters: int = 1
    rock_heat_capacity: float = 1.0   # factor on the bedrock/filler heat capacity (1 = physical)

    @property
    def horizon(self) -> float:
        return self.dt * self.n_steps


class Model:
    layout: StateLayout

    def run(self, x: np.ndarray, p: float, spec: MapSpec, tag: str = "run", want_diag: bool = False):
        """Return (x_out, diag) with diag a dict of extra fields (may be empty)."""
        raise NotImplementedError


def rms(v) -> float:
    v = np.asarray(v)
    return float(np.linalg.norm(v) / np.sqrt(max(v.size, 1)))


class FixedPointProblem:
    """F(y, p) = Phi(y s; p)/s - y in scaled variables, with finite-difference directional derivatives.

    fd_eps: rms size (scaled units) of the perturbation used for Jacobian-vector products.
    fd_scheme: "forward" (1 evaluation per product) or "central" (2).
    """

    def __init__(self, model: Model, spec: MapSpec, fd_eps: float = 1e-4, fd_scheme: str = "forward",
                 p_scale: float = 1.0):
        self.model = model
        self.spec = spec
        self.s = model.layout.scale_vector()
        self.n = len(self.s)
        self.fd_eps = float(fd_eps)
        self.fd_scheme = fd_scheme
        self.p_scale = float(p_scale)
        self.nfev = 0
        self.t_model = 0.0
        self.last_diag = {}

    # ---- conversions
    def scale(self, x):
        return np.asarray(x, float) / self.s

    def unscale(self, y):
        return np.asarray(y, float) * self.s

    # ---- map and residual
    def phi(self, y, p, tag="phi", want_diag=False, spec=None):
        t0 = time.time()
        xo, diag = self.model.run(self.unscale(y), p, spec or self.spec, tag=tag, want_diag=want_diag)
        self.t_model += time.time() - t0
        self.nfev += 1
        if want_diag:
            self.last_diag = diag
        yo = self.scale(xo)
        if not np.all(np.isfinite(yo)):
            raise ModelFailure("non-finite state returned")
        return yo

    def F(self, y, p, tag="F", want_diag=False):
        return self.phi(y, p, tag=tag, want_diag=want_diag) - y

    # ---- directional derivatives
    def dF(self, y, p, F0, dy, dp=0.0, tag="jv"):
        """(dF/dy) dy + (dF/dp) dp by finite differences; dp in physical parameter units."""
        nz = np.sqrt(rms(dy) ** 2 + (dp / self.p_scale) ** 2)
        if nz == 0:
            return np.zeros(self.n)
        eps = self.fd_eps / nz
        if self.fd_scheme == "central":
            Fp = self.F(y + eps * dy, p + eps * dp, tag=tag)
            Fm = self.F(y - eps * dy, p - eps * dp, tag=tag)
            return (Fp - Fm) / (2 * eps)
        Fp = self.F(y + eps * dy, p + eps * dp, tag=tag)
        return (Fp - F0) / eps

    def dphi(self, y, p, phi0, dy, spec=None, tag="arnoldi"):
        """Derivative of the map itself, D Phi dy (for stability analysis)."""
        nz = rms(dy)
        if nz == 0:
            return np.zeros(self.n)
        eps = self.fd_eps / nz
        if self.fd_scheme == "central":
            pp = self.phi(y + eps * dy, p, tag=tag, spec=spec)
            pm = self.phi(y - eps * dy, p, tag=tag, spec=spec)
            return (pp - pm) / (2 * eps)
        return (self.phi(y + eps * dy, p, tag=tag, spec=spec) - phi0) / eps
