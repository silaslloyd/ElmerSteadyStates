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

from .linalg import Recycler, gmres
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
    diag = getattr(prob, "last_diag", {}) or {}
    cx = next((v for k, v in diag.items() if k.lower() == "coordinate 1"), None)
    cz = next((v for k, v in diag.items() if k.lower() == "coordinate 3"), None)
    for b, sl in zip(lay.blocks, lay.slices().values()):
        db = d[sl]
        if len(db) == 0:
            continue
        i = int(np.argmax(np.abs(db)))
        g = int(b.gids[i])
        where = f"node {g}"
        if cx is not None:
            j = int(np.clip(np.searchsorted(cx[0], g), 0, len(cx[0]) - 1))
            if cx[0][j] == g:
                where += f" (x = {cx[1][j] / 1e3:.1f} km" + (f", z = {cz[1][j]:.0f} m)" if cz is not None else ")")
        out.append(f"{b.name}: rms {rms(db) * b.scale:.3g}, max {db[i] * b.scale:+.3g} at {where}")
    return "; ".join(out)


def newton_krylov(prob: FixedPointProblem, y0, p, opt: NewtonOptions = NewtonOptions(),
                  F0=None, checkpoint: str | None = None, label="newton") -> NewtonResult:
    y = np.asarray(y0, float).copy()
    F = prob.F(y, p, tag=f"{label}_F", want_diag=True) if F0 is None else F0
    nf = rms(F)
    hist, giters = [nf], []
    log.info(f"{label}: it 0  rms(F) = {nf:.3e}  (rate {nf / prob.spec.horizon:.3e} per unit time)  p = {p:.6g}")
    log.info(f"{label}: residual {describe_step(prob, F)}")
    nf_prev, eta = None, opt.eta_max
    # ESS_GMRES_RECYCLE=k (default off): GMRES deflated with the k harmonic Ritz vectors of the smallest |theta| of everything
    # the earlier solves computed (ess.linalg.Recycler): the slow modes (J ~ 0) cost the iterations, they are carried over from
    # Newton iteration to iteration and from point to point (while the state vector has the same size). No extra model runs.
    rec_k = int(os.environ.get("ESS_GMRES_RECYCLE", "0") or 0)
    rec = None
    if rec_k > 0:
        rec = getattr(prob, "_recycler", None)
        if rec is None or rec.k != rec_k:
            rec = prob._recycler = Recycler(rec_k)
    # ESS_NEWTON_RELAX=k (default off): after a failed line search, or an accepted step with lambda < ESS_NEWTON_RELAX_LAM
    # (default 0.25), up to k plain map steps, each kept only while rms(F) falls; the line search then stops after
    # ESS_NEWTON_RELAX_LS halvings (default 3) instead of [newton] linesearch
    relax_k = int(os.environ.get("ESS_NEWTON_RELAX", "0") or 0)
    relax_lam = float(os.environ.get("ESS_NEWTON_RELAX_LAM", "0.25"))
    relax_ls = int(os.environ.get("ESS_NEWTON_RELAX_LS", "3"))
    for it in range(1, opt.maxit + 1):
        if nf <= opt.tol:
            return NewtonResult(y, F, True, it - 1, hist, giters)
        if nf_prev is not None:                          # Eisenstat-Walker choice 2
            eta_new = 0.9 * (nf / nf_prev) ** 2
            if 0.9 * eta ** 2 > 0.1:
                eta_new = max(eta_new, 0.9 * eta ** 2)
            eta = eta_new
        eta = float(np.clip(eta, opt.eta_min, opt.eta_max))
        # no need to solve beyond the target. ESS_NEWTON_CAP_ETA=1: never looser than eta_max either (near the target
        # 0.5 tol / rms(F) reached 0.2-0.26 and Newton crept at ~20% per iteration, sweep 31889 d = 506.7 km; but there
        # GMRES needed 9 products for 0.2: F's noise floor may be near, and a tighter eta only runs GMRES into its
        # stagnation stop. A test (branch sweep) before it is the default.)
        eta = max(eta, 0.5 * opt.tol / max(nf, 1e-300))
        if os.environ.get("ESS_NEWTON_CAP_ETA") == "1":
            eta = min(eta, opt.eta_max)

        def Jv(v):
            return prob.dF(y, p, F, v, tag=f"{label}_jv")

        solve = rec.solve if rec is not None else gmres
        d, info = solve(Jv, -F, tol=eta, restart=opt.gmres_restart, maxiter=opt.gmres_maxiter,
                        callback=lambda k, r: log.debug(f"  gmres {k}: {r:.3e}"))
        giters.append(info.iterations)
        log.info(f"{label}: it {it}  gmres {info.iterations} its, rel. res {info.residual:.2e} "
                 f"(eta {eta:.1e}){'' if info.converged else ('  STAGNATED' if info.stalled else '  NOT CONVERGED')}")
        dmax = np.max(np.abs(d))
        log.info(f"{label}: step {describe_step(prob, d)}")
        if dmax > opt.max_step:
            d *= opt.max_step / dmax
            log.info(f"{label}: step capped (max |dy| {dmax:.3g} -> {opt.max_step})")

        lam, accepted = 1.0, False
        for _ in range((min(opt.linesearch, relax_ls) if relax_k else opt.linesearch) + 1):
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
        if accepted:
            y, F, nf_prev, nf = yt, Ft, nf, nt
            hist.append(nf)
            log.info(f"{label}: it {it}  rms(F) = {nf:.3e}  lambda = {lam:g}  max|dy| = {lam * np.max(np.abs(d)):.3g}  "
                     f"model runs so far {prob.nfev}")
        if relax_k and (not accepted or lam < relax_lam):
            # RELAXATION (ESS_NEWTON_RELAX=k): Newton's linear model fails on the stiff modes (the subtemperate
            # sliding's 0.15 K scale); plain map steps y <- Phi(y) = y + F let them decay. Each kept only if rms(F)
            # falls: a fast-growing mode raises it at once, and the relaxation stops (one map run lost).
            nr = 0
            for j in range(relax_k):
                yr = y + F
                try:
                    Fr = prob.F(yr, p, tag=f"{label}_relax", want_diag=True)
                    nrr = rms(Fr)
                except ModelFailure as e:
                    log.warning(f"{label}: relaxation step {j + 1} failed: {e}")
                    break
                if nrr >= nf:
                    log.info(f"{label}: relaxation step {j + 1}: rms(F) {nf:.3e} -> {nrr:.3e}: rises, discarded")
                    break
                log.info(f"{label}: relaxation step {j + 1}: rms(F) {nf:.3e} -> {nrr:.3e} (x {nrr / nf:.3f}), kept")
                y, F, nf = yr, Fr, nrr
                nr += 1
            if nr:
                nf_prev = None                       # Eisenstat-Walker restarts from eta_max
                hist.append(nf)
                log.info(f"{label}: it {it}  rms(F) = {nf:.3e}  after {nr} relaxation step(s)  model runs so far "
                         f"{prob.nfev}")
                eta = opt.eta_max
            elif not accepted:
                log.warning(f"{label}: line search failed, relaxation did not help")
                return NewtonResult(y, F, False, it, hist, giters)
        elif not accepted:
            log.warning(f"{label}: line search failed")
            return NewtonResult(y, F, False, it, hist, giters)
        if checkpoint:
            np.savez(checkpoint, y=y, p=p, it=it)
    return NewtonResult(y, F, nf <= opt.tol, opt.maxit, hist, giters)
