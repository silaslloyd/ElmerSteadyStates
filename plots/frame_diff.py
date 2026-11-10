#!/usr/bin/env python3
"""Where two frames of a frames file (plots/vtu_frames.py) differ most: surface / basal speed, vertical velocity and
surface elevation, the largest changes with their positions (x, y, r, angle) and grounded mask. For locating what a
failing solve did (the frame after it against the one before).

    python3 plots/frame_diff.py FRAMES.npz [K0 K1]       (default the last two frames)
"""
import sys

import numpy as np

a = sys.argv[1:]
if not a:
    raise SystemExit(__doc__)
F = np.load(a[0])
n = len(F["frame_times"])
k0, k1 = (int(a[1]), int(a[2])) if len(a) >= 3 else (n - 2, n - 1)
t0, t1 = F["frame_times"][k0], F["frame_times"][k1]
print(f"frames {k0} (t {t0:g}) -> {k1} (t {t1:g})")
for side in ("top", "base"):
    x, y = F[f"{side}_x"].astype(float), F[f"{side}_y"].astype(float)
    r, ph = np.hypot(x, y) / 1e3, np.degrees(np.arctan2(y, x))
    sp = lambda k: np.hypot(F[f"{side}_vx"][k], F[f"{side}_vy"][k]).astype(float)
    fields = {"speed": (sp(k0), sp(k1)), "vz": (F[f"{side}_vz"][k0].astype(float), F[f"{side}_vz"][k1].astype(float)),
              "z": (F[f"{side}_z"][k0].astype(float), F[f"{side}_z"][k1].astype(float))}
    m = F["base_mask"][k1] if side == "base" and "base_mask" in F.files else None
    for name, (v0, v1) in fields.items():
        d = v1 - v0
        o = np.argsort(-np.abs(np.nan_to_num(d)))[:8]
        print(f"\n{side} {name}: rms change {np.sqrt(np.nanmean(d ** 2)):.3g}, largest:")
        for i in o:
            print(f"   {d[i]:+10.3g} (from {v0[i]:9.3g})  x {x[i] / 1e3:7.2f} y {y[i] / 1e3:7.2f}  r {r[i]:7.2f} "
                  f"angle {ph[i]:+7.3f}" + (f"  mask {m[i]:+.0f}" if m is not None else ""))
