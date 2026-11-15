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
zk = "base_z" if "base_z" in d.files else "base_v"
zb, zs, gm = d[zk][k], d[zk.replace("base", "top")][k], d["base_mask"][k]
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
              f"Zb {np.nanmedian(zb[m]):8.1f}  implied sea level {np.nanmedian(sl[m]):8.1f} ({np.nanmin(sl[m]):.1f} .. {np.nanmax(sl[m]):.1f})"
              + "".join(f"  {n_} {np.nanmedian(d['base_' + n_][k][m]):.3g}" for n_ in ("cfl", "dz", "vz") if 'base_' + n_ in d.files)
              + (f"  mesh z - Zb {np.nanmedian(zb[m] - d['base_zbv'][k][m]):.1f}" if "base_zbv" in d.files else "")
              + (f"  speed {np.nanmedian(np.hypot(d['base_vx'][k][m], d['base_vy'][k][m])):.1f}" if "base_vx" in d.files else ""))
# across the sector (the lateral direction): the floating ice within 10 km of the front and in the band just past the
# full-Stokes strip, in angle bins
ph = np.degrees(np.arctan2(by, bx))
print("across the sector (angle bins; y at the front):")
for lo_, hi_ in ((R[fl].min(), R[fl].min() + 10), (R[fl].min() + 10, R[fl].min() + 25)):
    band = fl & (R >= lo_) & (R < hi_)
    if not band.any():
        continue
    e = np.linspace(ph[band].min(), ph[band].max(), 9)
    print(f"  r {lo_:.0f}-{hi_:.0f} km:")
    for a, b in zip(e[:-1], e[1:]):
        m = band & (ph >= a) & (ph <= b)
        if m.sum():
            yk = np.median(by[m]) / 1e3
            print(f"    angle {a:6.2f}..{b:6.2f} (y ~{yk:7.1f} km): n {m.sum():5d}  H {np.nanmedian(H[m]):6.1f}  "
                  f"Zs {np.nanmedian(zs_b[m]):7.1f}  Zb {np.nanmedian(zb[m]):7.1f}  implied sea level {np.nanmedian(sl[m]):7.1f}")
