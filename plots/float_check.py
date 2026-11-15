#!/usr/bin/env python3
"""The sea level the floating ice implies, Zb + (rho_i/rho_w) H, in 5 km bins of radius, at one frame of a
vtu_frames.py / job_frames.py npz (default the last; or the frame nearest T yr):

    python3 plots/float_check.py FRAMES.npz [T]

A freely floating column gives the sea level the run had at that time everywhere on the shelf (the full-Stokes
strip and the SSA shelf alike); a different value in one part means that part floats about another sea level.
"""
import numpy as np, sys
d = np.load(sys.argv[1]); t = float(sys.argv[2]) if len(sys.argv) > 2 else None
ft = d["frame_times"]; k = int(np.argmin(np.abs(ft - t))) if t is not None else len(ft) - 1
print("frame", k, "t", ft[k], "keys", sorted(d.files))
bx, by, tx, ty = d["base_x"], d["base_y"], d["top_x"], d["top_y"]
zb, zs, gm = d["base_v"][k], d["top_v"][k], d["base_mask"][k]
key = lambda x, y: np.round(x, 0) * 1e7 + np.round(y, 0)
kb, kt = key(bx, by), key(tx, ty)
o = np.argsort(kt); j = np.searchsorted(kt[o], kb); j = np.clip(j, 0, len(kt) - 1)
ok = kt[o][j] == kb
zs_b = np.where(ok, zs[o][j], np.nan)
r = 920 / 1003.27
H = zs_b - zb
sl = zb + r * H                      # the sea level a freely floating column implies
fl = ok & (gm < -0.5)
R = np.hypot(bx, by) / 1e3
print(f"matched {ok.sum()} of {len(bx)} base nodes; floating {fl.sum()}")
for a in range(int(R[fl].min()) // 5 * 5, int(R[fl].max()) + 5, 5):
    m = fl & (R >= a) & (R < a + 5)
    if m.sum():
        print(f"r {a:4d}-{a+5:<4d} km: n {m.sum():5d}  H {np.nanmedian(H[m]):7.1f}  Zs {np.nanmedian(zs_b[m]):8.1f}  "
              f"Zb {np.nanmedian(zb[m]):8.1f}  implied sea level {np.nanmedian(sl[m]):8.1f} ({np.nanmin(sl[m]):.1f} .. {np.nanmax(sl[m]):.1f})")
