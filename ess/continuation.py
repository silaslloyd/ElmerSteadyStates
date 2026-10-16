"""
Pseudo-arclength continuation of fixed points of the time-step map in one parameter p.

Unknowns z = (y, p~), p~ = p / p_scale, with the inner product <a, b> = a_y.b_y / n + a_p b_p
(state measured as an rms, so a branch step of ds moves the state by ~ds scaled units rms or the
parameter by ~ds p_scale). Each corrector step solves the bordered system

    [ J     F_p ] [dy ]     [ F ]
    [ t_y/n t_p ] [dp~] = - [ N ],    N = <t, z - z_k> - ds,

matrix-free with GMRES; one model run gives J dy + F_p dp at once (a directional difference in
(y, p)). Folds are passed without trouble, and located from the sign change of dp/ds.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field

import numpy as np

from .linalg import gmres
from .model import FixedPointProblem, ModelFailure, rms
from .newton import NewtonOptions, newton_krylov

log = logging.getLogger("ess")


@dataclass
class ContinuationOptions:
    ds: float = 0.5                # initial arclength step
    ds_min: float = 1e-3
    ds_max: float = 5.0
    dp0: float = 0.2               # first (natural-parameter) step, physical units, sign = direction
    p_min: float = -np.inf
    p_max: float = np.inf
    max_points: int = 50
    max_folds: int = 10
    tol: float = 1e-7              # rms(F) target
    maxit: int = 8                 # corrector iterations
    target_its: int = 3            # step adaptation: ds x2 below, x1.25 at, x target/its (>= 0.5) above
    eta: float = 1e-3              # smallest GMRES relative tolerance in the corrector (it starts at 0.1)
    gmres_restart: int = 40
    gmres_maxiter: int = 80
    max_step: float = 5.0          # cap on a corrector update (scaled units, max norm)
    contraction: float = 0.5       # give up a step when rms(F) falls by less than this factor per iteration


def wdot(prob, a, b):
    return float(a[:-1] @ b[:-1]) / prob.n + float(a[-1] * b[-1])


def wnorm(prob, a):
    return np.sqrt(max(wdot(prob, a, a), 0.0))


class BranchStore:
    """Points on disk: <dir>/points/pt_XXXX.npz (scaled y, p, arclength) and <dir>/branch.csv."""

    COLS = ["index", "s", "p", "rms_y", "mean_y", "rms_F", "newton_its", "gmres_its", "model_runs",
            "tangent_p", "ds", "fold"]

    def __init__(self, path, prob: FixedPointProblem, on_save=None):
        self.path = path
        self.prob = prob
        self.on_save = on_save        # f(index, y, p, row) -> dict of extra columns (and saves states)
        os.makedirs(os.path.join(path, "points"), exist_ok=True)
        self.csv = os.path.join(path, "branch.csv")
        self.rows = []
        if os.path.exists(self.csv):
            with open(self.csv) as f:
                self.rows = list(csv.DictReader(f))

    def point_file(self, k):
        return os.path.join(self.path, "points", f"pt_{k:04d}.npz")

    def __len__(self):
        return len(self.rows)

    def append(self, y, p, row):
        k = len(self.rows)
        row = dict(row, index=k)
        np.savez(self.point_file(k), y=y, p=p, s=row["s"])
        if self.on_save:
            row.update(self.on_save(k, y, p, row) or {})
        cols = self.COLS + [c for c in row if c not in self.COLS]
        if self.rows:
            cols += [c for c in self.rows[0] if c not in cols]
        self.rows.append({c: row.get(c, "") for c in cols})
        tmp = self.csv + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in self.rows:
                w.writerow({c: r.get(c, "") for c in cols})
        os.replace(tmp, self.csv)
        return k

    def load(self, k):
        with np.load(self.point_file(k)) as z:
            return z["y"].copy(), float(z["p"]), float(z["s"])


def continuation(prob: FixedPointProblem, y0, p0, store: BranchStore, opt: ContinuationOptions,
                 newton_opt: NewtonOptions | None = None):
    ps = prob.p_scale
    newton_opt = newton_opt or NewtonOptions(tol=opt.tol)
    nfold = sum(1 for r in store.rows if str(r.get("fold", "")) not in ("", "0", "False"))

    def row(y, p, s, F, its, gits, tp, ds, fold=0):
        return dict(s=s, p=p, rms_y=rms(y), mean_y=float(np.mean(y)), rms_F=rms(F), newton_its=its,
                    gmres_its=gits, model_runs=prob.nfev, tangent_p=tp, ds=ds, fold=fold)

    # ---- the first two points (natural-parameter Newton), unless resuming
    if len(store) == 0:
        r = newton_krylov(prob, y0, p0, newton_opt, label="newton p0")
        if not r.converged:
            raise RuntimeError("continuation: Newton did not converge at the starting point")
        store.append(r.y, p0, row(r.y, p0, 0.0, r.F, r.iterations, sum(r.gmres_iters), np.nan, opt.ds))
    if len(store) == 1:
        y0, p0, _ = store.load(0)
        p1 = p0 + opt.dp0
        r = newton_krylov(prob, y0, p1, newton_opt, label="newton p1")
        if not r.converged:
            raise RuntimeError("continuation: Newton did not converge at the second point; reduce dp0")
        s1 = np.sqrt(rms(r.y - y0) ** 2 + (opt.dp0 / ps) ** 2)
        store.append(r.y, p1, row(r.y, p1, s1, r.F, r.iterations, sum(r.gmres_iters), np.sign(opt.dp0), opt.ds))

    ya, pa, sa = store.load(len(store) - 2)
    yb, pb, sb = store.load(len(store) - 1)
    ds = float(store.rows[-1].get("ds") or opt.ds)
    if len(store) == 2:
        # first secant step: don't extrapolate much beyond the distance between the first two points
        ds = min(ds, max(2.0 * (sb - sa), opt.ds_min))
        log.info(f"continuation: first arclength step ds = {ds:.3g} (twice the first point spacing)")
    t_prev_p = None

    while len(store) < opt.max_points:
        zb = np.append(yb, pb / ps)
        # secant over a baseline of at least ~ds: the converged points carry errors along poorly
        # determined directions, and a secant over a short baseline extrapolated far is mostly noise
        za, base = np.append(ya, pa / ps), sb - sa
        k = len(store) - 3
        while base < ds and k >= 0:
            yk, pk, sk = store.load(k)
            zk = np.append(yk, pk / ps)
            if np.sign((zb - zk)[-1]) != np.sign((zb - za)[-1]):
                break                                   # do not reach back past a fold
            za, base = zk, sb - sk
            k -= 1
        if base > sb - sa:
            log.info(f"continuation: secant over a baseline of {base:.3g} (point {k + 1})")
        t = zb - za
        t /= wnorm(prob, t)
        if t_prev_p is None:
            t_prev_p = t[-1]

        ok, z, F, its, gits = _corrector(prob, zb, t, ds, opt)
        if not ok:
            ds *= 0.5
            log.info(f"continuation: corrector failed, ds -> {ds:.3g}")
            if ds < opt.ds_min:
                log.warning("continuation: ds below ds_min, stopping")
                break
            continue
        yn, pn = z[:-1], z[-1] * ps
        sn = sb + wnorm(prob, z - zb)
        tn_p = (z - zb)[-1]
        fold = int(np.sign(tn_p) != np.sign(t_prev_p) and tn_p != 0 and t_prev_p != 0)
        rw = row(yn, pn, sn, F, its, gits, tn_p / max(wnorm(prob, z - zb), 1e-300), ds, fold)
        if fold:
            nfold += 1
            pf = _fold_estimate([sa, sb, sn], [pa, pb, pn])
            rw["p_fold_estimate"] = pf
            log.info(f"continuation: FOLD between p = {pb:.6g} and {pn:.6g}  (quadratic estimate p_fold = {pf:.6g})")
        k = store.append(yn, pn, rw)
        log.info(f"continuation: point {k}  p = {pn:.6g}  s = {sn:.4g}  rms(F) = {rms(F):.2e}  "
                 f"its {its}  gmres {gits}  ds {ds:.3g}  model runs {prob.nfev}")
        ya, pa, sa, yb, pb, sb, t_prev_p = yb, pb, sb, yn, pn, sn, tn_p
        grow = 2.0 if its < opt.target_its else (1.25 if its == opt.target_its else max(opt.target_its / its, 0.5))
        ds = float(np.clip(ds * grow, opt.ds_min, opt.ds_max))
        store.rows[-1]["ds"] = ds
        if not (opt.p_min <= pn <= opt.p_max):
            log.info("continuation: parameter left [p_min, p_max], stopping")
            break
        if nfold >= opt.max_folds:
            log.info("continuation: max_folds reached, stopping")
            break
    return store


def _fold_estimate(s, p):
    try:
        a, b, c = np.polyfit(np.asarray(s) - s[1], p, 2)
        return float(c - b * b / (4 * a)) if a != 0 else float(p[1])
    except Exception:
        return float(np.nan)


def _corrector(prob, zb, t, ds, opt):
    n = prob.n
    ps = prob.p_scale
    sq = np.sqrt(n)
    z = zb + ds * t
    gits = 0
    try:
        F = prob.F(z[:-1], z[-1] * ps, tag="cont_F", want_diag=True)
    except ModelFailure as e:
        log.info(f"continuation: model failed at the predictor: {e}")
        return False, z, None, 0, 0
    nf_prev = np.inf
    for it in range(opt.maxit + 1):
        N = wdot(prob, t, z - zb) - ds
        nf = rms(F)
        log.info(f"  corrector it {it}: rms(F) = {nf:.3e}  N = {N:.2e}")
        if nf <= opt.tol and abs(N) <= max(opt.tol, 1e-10 * ds):
            return True, z, F, it, gits
        if it == opt.maxit:
            break
        if it >= 2 and nf > opt.contraction * nf_prev:
            log.info(f"  corrector: not contracting (factor {nf / nf_prev:.2f}), giving up on this step")
            break
        y, p = z[:-1], z[-1] * ps

        def mv(u, y=y, p=p, F=F):
            dy = u[:-1] * sq
            dpt = u[-1]
            top = prob.dF(y, p, F, dy, dp=dpt * ps, tag="cont_jv") / sq
            bot = float(t[:-1] @ u[:-1]) / sq + t[-1] * dpt
            return np.append(top, bot)

        rhs = -np.append(F / sq, N)
        # forcing term: loose while far from the branch, tighter as the residual falls (Eisenstat-Walker)
        eta = 0.1 if not np.isfinite(nf_prev) else 0.9 * (nf / nf_prev) ** 2
        eta = float(np.clip(eta, opt.eta, 0.1))
        eta = max(eta, 0.5 * opt.tol / max(nf, 1e-300))
        u, info = gmres(mv, rhs, tol=eta, restart=opt.gmres_restart, maxiter=opt.gmres_maxiter,
                        callback=lambda k, r: log.debug(f"  gmres {k}: {r:.3e}"))
        gits += info.iterations
        log.info(f"  corrector it {it + 1}: gmres {info.iterations} its, rel. res {info.residual:.2e} (eta {eta:.1e})"
                 f"{'' if info.converged else ('  STAGNATED' if info.stalled else '  NOT CONVERGED')}")
        dz = np.append(u[:-1] * sq, u[-1])
        from .newton import describe_step
        log.info(f"  corrector step: {describe_step(prob, dz[:-1])}; dp {dz[-1] * ps:+.3g}")
        dmax = np.max(np.abs(dz))
        if dmax > opt.max_step:
            dz *= opt.max_step / dmax
        lam = 1.0
        for _ in range(4):
            zt = z + lam * dz
            try:
                Ft = prob.F(zt[:-1], zt[-1] * ps, tag="cont_F", want_diag=True)
                if rms(Ft) < max(nf, opt.tol) * 1.0001 or lam < 0.2:
                    break
            except ModelFailure as e:
                log.info(f"  corrector: model failed ({e}), halving")
                Ft = None
            lam *= 0.5
        if Ft is None:
            return False, z, F, it, gits
        nf_prev = nf
        z, F = zt, Ft
    return False, z, F, opt.maxit, gits
