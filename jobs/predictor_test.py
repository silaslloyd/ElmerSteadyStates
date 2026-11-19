#!/usr/bin/env python3
"""Test of the sweep predictor (ess/predictor.py): from a converged sweep point (patch length d_old), the first guess for
d_new with the margin structure moved downstream with the head, against the plain previous state.

    python3 jobs/predictor_test.py CONFIG STATE.npz --d-new-km D [--core-km 10] [--taper-km T] [--newton]
                                   [--out-dir DIR]

1. the patch mask for d_new (as the sweep: temperate within d_new of the grounding line, floating nodes temperate);
2. F at the plain state (the sweep's start; a cache hit when the sweep has evaluated it) and at the shifted one: rms(F)
   and where the residual is, for both (2 Elmer runs at most);
3. --newton: Newton from the shifted state (the config's [newton]; masks frozen, no grounding-line rounds), to compare
   its iterations with the sweep's from the plain state.
Writes DIR/predicted.npz (the shifted state) and, with --newton, DIR/newton.npz (default DIR:
<work>/predictor_test_d<D>).
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("state")
    ap.add_argument("--d-new-km", type=float, required=True)
    ap.add_argument("--core-km", type=float, default=10.0)
    ap.add_argument("--taper-km", type=float)
    ap.add_argument("--newton", action="store_true")
    ap.add_argument("--maxit", type=int)
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    import logging
    from dataclasses import replace
    from ess.cli import _patch_diagnostics, _problem, _setup
    from ess.config import newton_options
    from ess.model import rms
    from ess.newton import describe_step, newton_krylov
    from ess.predictor import shift_margin
    from ess.spectrum import gl_position, patch_mask
    from ess.statefile import find_name, save_state
    log = logging.getLogger("ess")
    cfg, model, saved = _setup(SimpleNamespace(config=a.config, state=a.state, verbose=False))
    _patch_diagnostics(model)
    prob = _problem(cfg, model)
    p = saved.p
    if "sweep_d" not in saved.meta:
        raise SystemExit(f"{a.state}: not a sweep point (no meta sweep_d)")
    d_old, d_new = float(saved.meta["sweep_d"]), a.d_new_km * 1e3
    xg = gl_position(saved.frozen, saved.diag)
    h_old = xg - d_old
    delta = d_old - d_new
    out = a.out_dir or os.path.join(cfg["_work"], f"predictor_test_d{a.d_new_km:g}")
    os.makedirs(out, exist_ok=True)
    log.info(f"predictor test: {a.state} (d {d_old / 1e3:g} km, head {h_old / 1e3:.2f} km, grounding line "
             f"{xg / 1e3:.2f} km) -> d {d_new / 1e3:g} km (head {(xg - d_new) / 1e3:.2f} km)")
    kct = find_name(saved.frozen, "ColdTempMask")
    g_ct, ct, _ = patch_mask(saved.frozen, saved.diag, d_new)
    saved.frozen[kct] = (g_ct, ct)
    x_pred, info = shift_margin(saved, h_old, delta, core=a.core_km * 1e3,
                                taper=a.taper_km * 1e3 if a.taper_km else None, x_gl=xg)
    save_state(os.path.join(out, "predicted.npz"), saved.layout, x_pred, p, dict(saved.frozen), saved.diag,
               sweep_d=d_new, x_gl=xg, predictor=f"shift {delta / 1e3:g} km")
    res = {}
    for name, x in (("plain", saved.x), ("shifted", x_pred)):
        t0 = time.time()
        y = prob.scale(x)
        F = prob.F(y, p, tag=f"predictor_{name}", want_diag=True)
        res[name] = (y, F, prob.last_diag)
        log.info(f"predictor test: {name:7s} start: rms(F) = {rms(F):.3e} ({time.time() - t0:.0f} s); residual "
                 f"{describe_step(prob, F)}")
    r0, r1 = rms(res["plain"][1]), rms(res["shifted"][1])
    log.info(f"predictor test: rms(F) plain {r0:.3e}, shifted {r1:.3e}: x {r0 / r1:.2f} smaller with the shift")
    if a.newton:
        nopt = newton_options(cfg)
        if a.maxit:
            nopt = replace(nopt, maxit=a.maxit)
        y, F, _ = res["shifted"]
        r = newton_krylov(prob, y, p, nopt, F0=F, label=f"predictor d={d_new / 1e3:g} km")
        log.info(f"predictor test: Newton from the shifted state {'converged' if r.converged else 'did NOT converge'} "
                 f"in {r.iterations} its (rms {r.history[-1]:.3e}; model runs {prob.nfev})")
        if r.converged:
            save_state(os.path.join(out, "newton.npz"), saved.layout, prob.unscale(r.y), p, dict(saved.frozen),
                       prob.last_diag, sweep_d=d_new, x_gl=xg)
    log.info(f"predictor test: done ({out})")


if __name__ == "__main__":
    main()
