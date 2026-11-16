#!/usr/bin/env python3
"""Is a state periodic across y? Pairs every node on the side y = 0 with its image on y = W (same x, same position in the column) and
prints, per field, the largest mismatch and where it is (x, z), plus the mismatch along x (worst per 10 km bin), so a
stripe from one partition shows as one bad x range. Works on full and slim states (plots/slim_states.py).

  python3 periodic_test/seam_check.py STATE.npz [--fields Temperature "Velocity 1" ...]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import load_state, find_name

ap = argparse.ArgumentParser()
ap.add_argument("state")
ap.add_argument("--fields", nargs="*", default=["Temperature", "Velocity 1", "Velocity 2", "Velocity 3", "Zs", "Zb",
                                                 "ColdTempMask", "GroundedMask", "Water Sheet Thickness"])
ap.add_argument("--width", type=float, default=None, help="the periodic translation (m); default: found from the sides")
a = ap.parse_args()
s = load_state(a.state)
allf = dict(s.diag); allf.update(s.frozen)
blocks = s.layout.split(s.x)
for b in s.layout.blocks:
    allf[b.name] = (b.gids, blocks[b.name])
get = lambda n: allf[find_name(allf, n)]
g, X = get("coordinate 1"); Y = get("coordinate 2")[1]; Z = get("coordinate 3")[1]
ymin, ymax = Y.min(), Y.max(); eps = 1e-6 * (ymax - ymin)
if ymin < -1.0 and -ymin < 0.25 * abs(ymax):
    # a strip with JAGGED periodic sides (configs/coarse_rect_ext.toml): side 1 the columns whose (x, y + W) is a
    # column too, side 2 those; W the translation (the most common y offset between columns at the same x)
    key = lambda xx, yy: (round(float(xx), 1), round(float(yy), 1))
    colz = {}
    for x, y, z, gg in zip(X, Y, Z, g):
        colz.setdefault(key(x, y), []).append((z, gg))
    W = a.width
    if W is None:
        lo_ = [k for k in colz if k[1] < ymin + 5e3]
        byx = {}
        for k in colz:
            if k[1] > ymax - 5e3:
                byx.setdefault(k[0], []).append(k[1])
        d_ = [yh - k[1] for k in lo_ for yh in byx.get(k[0], [])]
        v_, c_ = np.unique(np.round(d_, 0), return_counts=True)
        W = float(v_[np.argmax(c_)])
    pairs = []
    for (x, y), v in colz.items():
        w = colz.get(key(x, y + W))
        if w is not None and len(w) == len(v):
            pairs += [(p_[1], q_[1], x, p_[0]) for p_, q_ in zip(sorted(v), sorted(w))]
    print(f"{a.state}: jagged sides, translation {W:.0f} m, paired {len(pairs)} nodes")
else:
    lo, hi = np.abs(Y - ymin) < eps, np.abs(Y - ymax) < eps
    def cols(m):          # x -> the side's nodes at that x, bottom to top (z may differ if the sides do)
        c = {}
        for x, z, gg in zip(X[m], Z[m], g[m]):
            c.setdefault(round(x, 1), []).append((z, gg))
        return {k: sorted(v) for k, v in c.items()}
    cl, ch = cols(lo), cols(hi)
    pairs = [(a_[1], b_[1], x, a_[0]) for x in cl if x in ch and len(cl[x]) == len(ch[x]) for a_, b_ in zip(cl[x], ch[x])]
    print(f"{a.state}: width {ymax - ymin:.0f} m, side nodes {lo.sum()} / {hi.sum()}, paired {len(pairs)}")
if not pairs:
    raise SystemExit("no pairs (z differs between the sides? a slim state of a different mesh?)")
gl = np.array([p[0] for p in pairs]); gh = np.array([p[1] for p in pairs]); px = np.array([p[2] for p in pairs])
pz = np.array([p[3] for p in pairs])
for n in a.fields:
    try:
        gg, v = get(n)
    except KeyError:
        continue
    d = dict(zip(gg.tolist(), np.asarray(v, float).tolist()))
    ok = np.array([a_ in d and b_ in d for a_, b_ in zip(gl, gh)])
    if not ok.any():
        continue
    va = np.array([d[x] for x in gl[ok]]); vb = np.array([d[x] for x in gh[ok]])
    diff = np.abs(va - vb); k = int(np.argmax(diff))
    scale = np.nanmax(np.abs(va)) or 1.0
    print(f"  {n:12s}: max |y=0 - y=W| {diff[k]:.4g} (of max |value| {scale:.4g}) at x {px[ok][k] / 1e3:.1f} km, "
          f"z {pz[ok][k]:.0f} m ({va[k]:.4g} vs {vb[k]:.4g}); median {np.median(diff):.3g}")
    xb = (px[ok] // 10000).astype(int)
    worst = {b: diff[xb == b].max() for b in np.unique(xb)}
    bad = sorted(worst.items(), key=lambda t: -t[1])[:4]
    print("      worst 10 km bins: " + ", ".join(f"{b * 10}-{b * 10 + 10} km {w:.3g}" for b, w in bad))
