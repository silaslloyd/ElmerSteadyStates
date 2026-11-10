#!/usr/bin/env python3
"""A sector run's centre line against the ice either side of it, per frame (plots/vtu_frames.py file): where the
surface sits lower / higher on the centre line (y = 0, the mirror line of the mesh) than 2-6 km off it at the same
radius, the thinnest ice, and the fastest surface speed. A trough or spike growing on the centre line shows here.

    python3 plots/sector_centre.py FRAMES.npz [--r R0 R1]      (km, default the whole sector)
"""
import sys

import numpy as np

a = sys.argv[1:]
if not a:
    raise SystemExit(__doc__)
F = np.load(a[0])
x, y = F["top_x"].astype(float), F["top_y"].astype(float)
r = np.hypot(x, y) / 1e3
r0, r1 = (float(a[a.index("--r") + 1]), float(a[a.index("--r") + 2])) if "--r" in a else (0, 1e9)
bx, by = F["base_x"].astype(float), F["base_y"].astype(float)
bi = dict(zip(zip(np.round(bx), np.round(by)), range(len(bx))))
ib = np.array([bi.get(k, -1) for k in zip(np.round(x), np.round(y))])
cen = (np.abs(y) < 300) & (r >= r0) & (r <= r1)
off = (np.abs(y) > 2e3) & (np.abs(y) < 6e3) & (r >= r0) & (r <= r1)
rb = np.round(r).astype(int)
for k, t in enumerate(F["frame_times"]):
    zs = F["top_z"][k].astype(float)
    sp = np.hypot(F["top_vx"][k], F["top_vy"][k]).astype(float)
    H = np.where(ib >= 0, zs - F["base_z"][k][np.maximum(ib, 0)], np.nan)
    offm = {}
    for b in np.unique(rb[off]):
        offm[b] = np.mean(zs[off & (rb == b)])
    d = np.array([zs[i] - offm.get(rb[i], np.nan) for i in np.where(cen)[0]])
    rc = r[cen]
    j, jj = np.nanargmin(d), np.nanargmax(d)
    h = np.where(cen, H, np.nan)
    m = np.nanargmin(np.where(r > 50, H, np.nan))
    print(f"t {t:7.0f}: centre line minus 2-6 km off it: lowest {d[j]:7.1f} m at r {rc[j]:6.1f}, highest {d[jj]:6.1f} m "
          f"at r {rc[jj]:6.1f} km | thinnest grounded-or-not ice (r > 50 km) {H[m]:6.0f} m at r {r[m]:6.1f} y {y[m] / 1e3:5.1f} "
          f"| fastest surface {np.nanmax(sp):7.0f} m/yr at r {r[np.nanargmax(sp)]:6.1f} y {y[np.nanargmax(sp)] / 1e3:5.1f}")
