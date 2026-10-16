"""
Newton-Krylov for fixed points of the time-step map (Tuckerman's method):

    F(y) = Phi_T(y) - y = 0,     J = D Phi_T - I.

J is never formed: J v = [F(y + eps v) - F(y)] / eps costs one model run. The time stepper acts as
the preconditioner: fast-decaying modes have D Phi_T ~ 0, so J ~ -I on them and GMRES only works on
the slow (and the unstable) modes. Newton converges to unstable steady states as readily as to
stable ones, which is the point.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from .linalg import gmres
from .model import FixedPointProblem, ModelFailure, rms

log = logging.getLogger("ess")


@dataclass
class NewtonOptions:
    tol: float = 1e-7              # rms(F) in scaled units (over one map horizon)
    maxit: int = 15
    eta_max: float = 0.1           # Eisenstat-Walker forcing term bounds
    eta_min: float = 1e-5
    gmres_restart: int = 40
    gmres_maxiter: int = 80
    max_step: float = 5.0          # cap on max |dy| (scaled units) per Newton step
    linesearch: int = 6            # max step halvings
    armijo: float = 1e-4


@dataclass
class NewtonResult:
    y: np.ndarray
    F: np.ndarray
    converged: bool
    iterations: int
    history: list = field(default_factory=list)   # rms(F) per iteration
    gmres_iters: list = field(default_factory=list)


def describe_step(prob, d) -> str:
    """Where is a (scaled) step large? Max |d| per state block, in physical units, with the node."""
    out = []
    lay = getattr(prob.model, "layout", None)
    if lay is None:
        return f"max |dy| {np.max(np.abs(d)):.3g}"
    for b, sl in zip(lay.blocks, lay.slices().values()):
        db = d[sl]
        if len(db) == 0:
            continue
        i = int(np.argmax(np.abs(db)))
        out.append(f"{b.name}: rms {rms(db) * b.scale:.3g}, max {db[i] * b.scale:+.3g} at node {int(b.gids[i])}")
    return "; ".join(out)


def newton_krylov(prob: FixedPointProblem, y0, p, opt: NewtonOptions = NewtonOptions(),
                  F0=None, checkpoint: str | None = None, label="newton") -> NewtonResult:
    y = np.asarray(y0, float).copy()
    F = prob.F(y, p, tag=f"{label}_F", want_diag=True) if F0 is None else F0
    nf = rms(F)
    hist, giters = [nf], []
    log.info(f"{label}: it 0  rms(F) = {nf:.3e}  (rate {nf / prob.spec.horizon:.3e} per unit time)  p = {p:.6g}")
    nf_prev, eta = None, opt.eta_max
    for it in range(1, opt.maxit + 1):
        if nf <= opt.tol:
            return NewtonResult(y, F, True, it - 1, hist, giters)
        if nf_prev is not None:                          # Eisenstat-Walker choice 2
            eta_new = 0.9 * (nf / nf_prev) ** 2
            if 0.9 * eta ** 2 > 0.1:
                eta_new = max(eta_new, 0.9 * eta ** 2)
            eta = eta_new
        eta = float(np.clip(eta, opt.eta_min, opt.eta_max))
        eta = max(eta, 0.5 * opt.tol / max(nf, 1e-300))  # no need to solve beyond the target

        def Jv(v):
            return prob.dF(y, p, F, v, tag=f"{label}_jv")

        d, info = gmres(Jv, -F, tol=eta, restart=opt.gmres_restart, maxiter=opt.gmres_maxiter,
                        callback=lambda k, r: log.debug(f"  gmres {k}: {r:.3e}"))
        giters.append(info.iterations)
        log.info(f"{label}: it {it}  gmres {info.iterations} its, rel. res {info.residual:.2e} "
                 f"(eta {eta:.1e}){'' if info.converged else '  NOT CONVERGED'}")
        dmax = np.max(np.abs(d))
        log.info(f"{label}: step {describe_step(prob, d)}")
        if dmax > opt.max_step:
            d *= opt.max_step / dmax
            log.info(f"{label}: step capped (max |dy| {dmax:.3g} -> {opt.max_step})")

        lam, accepted = 1.0, False
        for _ in range(opt.linesearch + 1):
            yt = y + lam * d
            try:
                Ft = prob.F(yt, p, tag=f"{label}_F", want_diag=True)
                nt = rms(Ft)
            except ModelFailure as e:
                log.warning(f"{label}: model failed at lambda = {lam:g}: {e}")
                nt = np.inf
            if nt <= (1 - opt.armijo * lam) * nf:
                accepted = True
                break
            log.info(f"{label}: lambda = {lam:g} rejected (rms F {nt:.3e})")
            lam *= 0.5
        if not accepted:
            log.warning(f"{label}: line search failed")
            return NewtonResult(y, F, False, it, hist, giters)
        y, F, nf_prev, nf = yt, Ft, nf, nt
        hist.append(nf)
        log.info(f"{label}: it {it}  rms(F) = {nf:.3e}  lambda = {lam:g}  max|dy| = {lam * np.max(np.abs(d)):.3g}  "
                 f"model runs so far {prob.nfev}")
        if checkpoint:
            np.savez(checkpoint, y=y, p=p, it=it)
    return NewtonResult(y, F, nf <= opt.tol, opt.maxit, hist, giters)
