#!/usr/bin/env python3
"""Seam check of a sector run (plots/vtu_frames.py frames file): along the two angled sides against the centre line,
per 50 km of radius and per frame, the grounded mask, the cold/temperate mask, the base temperature and the surface
elevation. On a clean run the three columns agree (the sector is statistically the same at every angle; a real
pattern across the flow shows on one side too, but not on both sides alone).

    python3 plots/sector_seam_check.py FRAMES.npz [--frame K]     (default: every frame, summary per frame)
"""
import sys

import numpy as np

a = sys.argv[1:]
if not a:
    raise SystemExit(__doc__)
F = np.load(a[0])
ks = range(len(F["frame_times"]))
if "--frame" in a:
    ks = [int(a[a.index("--frame") + 1])]
x, y = F["base_x"].astype(float), F["base_y"].astype(float)
r, ph = np.hypot(x, y) / 1e3, np.arctan2(y, x)
s1, s2 = ph < ph.min() + 1e-6, ph > ph.max() - 1e-6
near = (~s1) & (~s2) & (np.abs(ph) > ph.max() - 2.5 / np.maximum(r, 1))          # within ~2.5 km of a side
cen = np.abs(ph) < 1.0 / np.maximum(r, 1)                                         # within ~1 km of the centre line
tx, ty = F["top_x"].astype(float), F["top_y"].astype(float)
zt = dict(zip(zip(np.round(tx), np.round(ty)), range(len(tx))))
it = np.array([zt.get(k, -1) for k in zip(np.round(x), np.round(y))])
bins = np.arange(0, r.max() + 50, 50)
print(f"{a[0]}: {s1.sum()} / {s2.sum()} side nodes, {near.sum()} next to the sides, {cen.sum()} on the centre line")
for k in ks:
    t = F["frame_times"][k]
    m = F["base_mask"][k]
    ct = F["base_ct"][k] if "base_ct" in F.files else np.full(len(x), np.nan)
    T = F["base_T"][k] if "base_T" in F.files else np.full(len(x), np.nan)
    zs = np.where(it >= 0, F["top_z"][k][np.maximum(it, 0)], np.nan)
    print(f"\nt = {t:g} yr   r bin (km) | grounded fraction side1 side2 near centre | temperate fraction (same) | "
          f"base T side1 side2 centre | Zs side1 side2 centre")
    for b0, b1 in zip(bins[:-1], bins[1:]):
        sel = [g & (r >= b0) & (r < b1) for g in (s1, s2, near, cen)]
        if min(sel[0].sum(), sel[1].sum(), sel[3].sum()) == 0:
            continue
        gf = [np.mean(m[s] > -0.5) if s.any() else np.nan for s in sel]
        tf = [np.mean(ct[s] > 0) if s.any() else np.nan for s in sel]
        tt = [np.nanmean(T[s]) for s in (sel[0], sel[1], sel[3])]
        zz = [np.nanmean(zs[s]) for s in (sel[0], sel[1], sel[3])]
        flag = "  <--" if max(abs(gf[0] - gf[3]), abs(gf[1] - gf[3]), abs(tf[0] - tf[3]), abs(tf[1] - tf[3])) > 0.5 else ""
        print(f"  {b0:4.0f}-{b1:4.0f} | {gf[0]:.2f} {gf[1]:.2f} {gf[2]:.2f} {gf[3]:.2f} | {tf[0]:.2f} {tf[1]:.2f} "
              f"{tf[2]:.2f} {tf[3]:.2f} | {tt[0]:7.2f} {tt[1]:7.2f} {tt[2]:7.2f} | {zz[0]:6.0f} {zz[1]:6.0f} {zz[2]:6.0f}{flag}")
