#!/usr/bin/env python3
"""The marine template's thin "filler" body (between bed - minh and the ice base Zb; heat and the mesh update are its
only solvers) and the ice column, per base node of a state: the filler thickness Zb - (bed - minh) (inverted where
< 0: negative element volumes in the heat solve), the ice thickness Zs - Zb, and where the worst ones are.

  python3 periodic_test/filler_check.py STATE.npz [STATE2.npz ...] [--minh 50]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import load_state, find_name
from ess.masks import _on

ap = argparse.ArgumentParser()
ap.add_argument("states", nargs="+")
ap.add_argument("--minh", type=float, default=50.0, help="[lua] minh of the config (m)")
a = ap.parse_args()
q = lambda v: " ".join(f"{x:.1f}" for x in np.percentile(v, [0, 1, 50]))
for path in a.states:
    s = load_state(path)
    allf = dict(s.diag); allf.update(s.frozen)
    blk = s.layout.split(s.x)
    for b in s.layout.blocks:
        allf[b.name] = (b.gids, blk[b.name])
    get = lambda n: allf[find_name(allf, n)]
    gb, zb = get("Zb")
    bed = _on(*get("bedrock"), gb)
    gm = _on(*get("GroundedMask"), gb)
    try:
        gx, X = get("coordinate 1"); X = _on(gx, X, gb); Y = _on(gx, get("coordinate 2")[1], gb)
    except KeyError:
        X = Y = np.full(len(gb), np.nan)
    fill = zb - (bed - a.minh)
    print(f"{path}: {len(gb)} base nodes (grounded {int((gm > -0.5).sum())}, floating {int((gm < -0.5).sum())})")
    print(f"  filler thickness Zb - (bed - {a.minh:g}) (m; min 1% median): {q(fill[np.isfinite(fill)])};"
          f" inverted (< 0): {int((fill < 0).sum())}, < 1 m: {int((fill < 1).sum())}, < 10 m: {int((fill < 10).sum())}")
    d = zb - bed
    print(f"  Zb - bed: grounded (m; min 1% median) {q(d[gm > -0.5])}; floating {q(d[gm < -0.5])};"
          f" floating below the bed: {int(((gm < -0.5) & (d < 0)).sum())}")
    for k in np.argsort(fill)[:5]:
        print(f"     x {X[k] / 1e3:.2f} km, y {Y[k] / 1e3:.2f} km: filler {fill[k]:.2f} m, Zb {zb[k]:.1f}, bed {bed[k]:.1f}, "
              f"mask {gm[k]:+.0f}")
    try:                                         # the surface node above each base node (same x, y)
        gs, zsv = get("Zs")
        Xs = _on(gx, get("coordinate 1")[1], gs); Ys = _on(gx, get("coordinate 2")[1], gs)
        top = {(round(x, 0), round(y, 0)): z for x, y, z in zip(Xs, Ys, zsv)}
        zs = np.array([top.get((round(x, 0), round(y, 0)), np.nan) for x, y in zip(X, Y)])
    except (KeyError, NameError):
        zs = None
    if zs is not None and np.isfinite(zs).any():
        h = zs - zb
        print(f"  ice thickness Zs - Zb (m; min 1% median): {q(h[np.isfinite(h)])}; < 10 m: {int((h < 10).sum())}")
