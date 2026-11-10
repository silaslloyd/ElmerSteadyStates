#!/usr/bin/env python3
"""The nodes near a point in every frame of a frames file (plots/vtu_frames.py): position, grounded mask, surface and
base elevation, surface and basal speed, vertical velocity. For a spot that keeps showing up in the step check.

    python3 plots/inspect_point.py FRAMES.npz X_KM Y_KM [RADIUS_KM, default 3] [--frames K0 K1]
"""
import sys

import numpy as np

a = sys.argv[1:]
if len(a) < 3:
    raise SystemExit(__doc__)
F = np.load(a[0])
px, py = float(a[1]) * 1e3, float(a[2]) * 1e3
rad = float(a[3]) * 1e3 if len(a) > 3 and not a[3].startswith("--") else 3e3
ks = range(len(F["frame_times"]))
if "--frames" in a:
    i = a.index("--frames")
    ks = range(int(a[i + 1]), int(a[i + 2]) + 1)
bx, by = F["base_x"].astype(float), F["base_y"].astype(float)
tx, ty = F["top_x"].astype(float), F["top_y"].astype(float)
ti = dict(zip(zip(np.round(tx), np.round(ty)), range(len(tx))))
near = np.where(np.hypot(bx - px, by - py) < rad)[0]
near = near[np.argsort(np.hypot(bx[near] - px, by[near] - py))]
it = [ti.get((round(bx[n]), round(by[n])), -1) for n in near]
print(f"{len(near)} base nodes within {rad / 1e3:g} km of ({px / 1e3:g}, {py / 1e3:g}) km; angle "
      f"{np.degrees(np.arctan2(by[near], bx[near])).round(3).tolist()[:12]} deg")
for k in ks:
    t = F["frame_times"][k]
    print(f"t {t:g}:   x      y   mask    Zs     Zb   |u_s|   |u_b|  vz_s")
    for n, j in zip(near, it):
        us = np.hypot(F["top_vx"][k][j], F["top_vy"][k][j]) if j >= 0 else np.nan
        ub = np.hypot(F["base_vx"][k][n], F["base_vy"][k][n])
        zs = F["top_z"][k][j] if j >= 0 else np.nan
        vz = F["top_vz"][k][j] if j >= 0 else np.nan
        print(f"     {bx[n] / 1e3:7.2f} {by[n] / 1e3:6.2f} {F['base_mask'][k][n]:5.0f} {zs:6.0f} {F['base_z'][k][n]:6.0f} "
              f"{us:7.0f} {ub:7.0f} {vz:6.1f}")
