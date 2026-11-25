#!/usr/bin/env python3
"""thin_rate.py

Where does the ice change between two states of the same run? Prints, in x bins, the mean thickness of each state, the change,
and the rate (m per 1000 yr), plus both grounding lines. Thinning confined to the last few km before the grounding line is a
local (boundary-layer) adjustment; thinning along the whole grounded profile means the model is not the one the start state
was built for (friction, rate factor, accumulation ...), and the GL just follows the interior down.

Uses only Zs, Zb, the coordinates and GroundedMask (no velocity), so the branch's start.npz works.

Usage (repo root):
    python3 diag/thin_rate.py work/gl_ssa/branch_ssa125/start.npz work/gl_ssa/branch_ssa125/transient.npz --years 6000
    options: --bin-km 20  --repo .
"""
import argparse
import os
import sys

import numpy as np


def columns(path, repo):
    sys.path.insert(0, repo)
    from ess.masks import _on
    from ess.statefile import find_name, load_state
    s = load_state(path)
    d = lambda n: s.diag[find_name(s.diag, n)]
    gx, X = d("coordinate 1")
    parts = s.layout.split(np.asarray(s.x))
    blk = lambda name: next(b for b in s.layout.blocks if b.name.lower() == name)
    key = lambda v: np.round(v, 1)

    def per_x(gids, vals):
        x = key(_on(gx, X, gids))
        vals = np.asarray(vals, float)
        ok = np.isfinite(x) & np.isfinite(vals)
        ux, inv = np.unique(x[ok], return_inverse=True)
        return dict(zip(ux, np.bincount(inv, weights=vals[ok]) / np.bincount(inv)))

    zsb, zbb = blk("zs"), blk("zb")
    zs, zb = per_x(zsb.gids, parts[zsb.name]), per_x(zbb.gids, parts[zbb.name])
    gm = per_x(*s.frozen[find_name(s.frozen, "GroundedMask")])
    xs = np.array(sorted(set(zs) & set(zb) & set(gm)))
    H = np.array([zs[v] - zb[v] for v in xs])
    g = np.array([gm[v] for v in xs])
    return xs, H, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="the earlier state (e.g. start.npz)")
    ap.add_argument("b", help="the later state (e.g. transient.npz)")
    ap.add_argument("--years", type=float, default=None, help="time between the two states, for the rate column")
    ap.add_argument("--bin-km", type=float, default=20.0)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    xa, Ha, ga = columns(args.a, repo)
    xb, Hb, gb = columns(args.b, repo)
    common = np.intersect1d(xa, xb)
    ia, ib = np.isin(xa, common), np.isin(xb, common)
    x, Ha, Hb, ga, gb = common, Ha[ia], Hb[ib], ga[ia], gb[ib]
    gla, glb = x[ga > -0.5].max(), x[gb > -0.5].max()
    print("grounding line: %.3f km (%s)  ->  %.3f km (%s)   %+.3f km" % (gla / 1e3, os.path.basename(args.a), glb / 1e3,
                                                                          os.path.basename(args.b), (glb - gla) / 1e3))
    print("%d common columns" % len(x))
    rate = (lambda d: "%9.2f" % (d / args.years * 1e3)) if args.years else (lambda d: "%9s" % "-")
    print("\n%15s %9s %9s %9s %9s" % ("x range (km)", "H a (m)", "H b (m)", "dH (m)", "m/kyr"))
    lo = 0.0
    top = max(gla, glb) + 5e3
    w = args.bin_km * 1e3
    while lo < top:
        m = (x >= lo) & (x < lo + w)
        if m.any():
            d = float(np.mean(Hb[m] - Ha[m]))
            print("%7.0f-%-7.0f %9.1f %9.1f %9.2f %s" % (lo / 1e3, (lo + w) / 1e3, np.mean(Ha[m]), np.mean(Hb[m]), d, rate(d)))
        lo += w
    print("\nlast 20 km before the GL, 2 km bins:")
    top2 = glb
    for k in range(10):
        hi = top2 - 2e3 * k
        m = (x > hi - 2e3) & (x <= hi)
        if m.any():
            d = float(np.mean(Hb[m] - Ha[m]))
            print("%7.1f-%-7.1f %9.1f %9.1f %9.2f %s" % ((hi - 2e3) / 1e3, hi / 1e3, np.mean(Ha[m]), np.mean(Hb[m]), d, rate(d)))


if __name__ == "__main__":
    main()
