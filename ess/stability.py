"""
Linear stability of a steady state from the time-step map.

Arnoldi on the linearised map M = D Phi_T(x*) (one model run per product). For an accurate time
stepper the eigenvalues are mu = exp(sigma T), so the dominant (largest |mu|) Ritz values are the least
stable modes, sigma = log(mu) / T. Backward Euler with step dt instead gives mu = (1 - sigma dt)^-n,
inverted as sigma_be = (1 - mu^(-1/n)) / dt; the two agree when |sigma| dt << 1.

Use a map with |sigma| dt << 1 for the modes of interest ([stability_map] in the config): backward
Euler is A-stable to a fault -- a mode with sigma dt > 2 maps to |mu| < 1 and would be reported stable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .linalg import arnoldi, ritz
from .model import FixedPointProblem, MapSpec, rms

log = logging.getLogger("ess")


@dataclass
class StabilityOptions:
    m: int = 30          # Krylov dimension (= model runs)
    nev: int = 10        # eigenvalues to report
    tol: float = 1e-3    # Ritz residual / |mu| below which a pair counts as converged
    seed: int = 0


@dataclass
class StabilityResult:
    mu: np.ndarray
    sigma: np.ndarray
    sigma_be: np.ndarray
    residual: np.ndarray
    converged: np.ndarray
    vectors: np.ndarray       # (n, nev) scaled
    drift: float              # rms(Phi(x*) - x*) in scaled units

    @property
    def n_unstable(self):
        return int(np.sum((np.abs(self.mu) > 1) & self.converged))


def stability(prob: FixedPointProblem, y_star, p, spec: MapSpec, opt: StabilityOptions = StabilityOptions(),
              checkpoint: str | None = None) -> StabilityResult:
    phi0 = prob.phi(y_star, p, tag="stab_base", spec=spec, want_diag=True)
    drift = rms(phi0 - y_star)
    log.info(f"stability: rms(Phi(x*) - x*) = {drift:.2e} over T = {spec.horizon:g} "
             f"(fd perturbation {prob.fd_eps:.1e}); m = {opt.m}")
    if drift > prob.fd_eps:
        log.warning("stability: the state is not a fixed point to within the finite-difference perturbation")

    def mv(v):
        return prob.dphi(y_star, p, phi0, v, spec=spec, tag="stab_jv")

    rng = np.random.default_rng(opt.seed)
    v0 = rng.standard_normal(prob.n)

    def cb(st):
        mu, _, res = ritz(st)
        k = min(3, len(mu))
        log.info(f"stability: step {st.k}: leading |mu| " +
                 ", ".join(f"{abs(m):.6f} (res {r:.1e})" for m, r in zip(mu[:k], res[:k])))

    import hashlib
    key = hashlib.sha1(np.ascontiguousarray(y_star, float).tobytes()
                       + repr((float(p), spec, prob.fd_eps, prob.fd_scheme, opt.seed)).encode()).hexdigest()
    st = arnoldi(mv, v0, opt.m, checkpoint=checkpoint, callback=cb, key=key)
    mu, X, res = ritz(st)
    k = min(opt.nev, len(mu))
    mu, X, res = mu[:k], X[:, :k], res[:k]
    T, n, dt = spec.horizon, spec.n_steps, spec.dt
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.log(mu.astype(complex)) / T
        sigma_be = (1 - mu.astype(complex) ** (-1.0 / n)) / dt
    conv = res <= opt.tol * np.maximum(np.abs(mu), 1e-300)
    for i in range(k):
        log.info(f"  mode {i}: mu = {mu[i]:.6f}  sigma = {sigma[i].real:+.5e} {sigma[i].imag:+.4e}i  "
                 f"(BE: {sigma_be[i].real:+.5e})  res {res[i]:.1e}{'' if conv[i] else '  (not converged)'}")
    return StabilityResult(mu, sigma, sigma_be, res, conv, X, drift)
