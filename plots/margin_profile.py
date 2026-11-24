#!/usr/bin/env python3
"""
Where the ice margin is and how much the sheet still changes, from a frames file (plots/vtu_frames.py): one line per
saved time, along the first row of nodes across the strip (y = min; with no friction dip the strip is y-uniform).

    python3 plots/margin_profile.py work/land_flowline_3d/fssa_spinup_progress_frames.npz [--thick 100] [--thin 35]

Per frame: the maximum ice thickness H = Zs - Zb and its x, the margin x_100 (the last x with H > --thick, default
100 m: the sheet) and x_thin (the last x with H > --thin, default 35 m: the thin layer beyond the margin keeps
H ~ 30 m, ess_Hmin), and the change since the previous frame: the maximum and the median of |dZs| over all surface
nodes and the maximum change of the margin x_100. A sheet that has settled: the changes go to ~0 and x_100 stops moving.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("--thick", type=float, default=100.0, help="m, thickness of the sheet's margin")
    ap.add_argument("--thin", type=float, default=35.0, help="m, thickness of the thin layer beyond it")
    a = ap.parse_args()
    d = np.load(a.frames, allow_pickle=True)
    t = np.asarray(d["frame_times"], float)
    tx, ty, tz = d["top_x"], d["top_y"], d["top_z"]
    bx, by, bz = d["base_x"], d["base_y"], d["base_z"]
    row = np.where(np.abs(ty - ty.min()) < 1e-6 * max(1.0, np.abs(ty).max()) + 1e-3)[0]
    row = row[np.argsort(tx[row])]
    x = tx[row]
    brow = np.where(np.abs(by - by.min()) < 1e-6 * max(1.0, np.abs(by).max()) + 1e-3)[0]
    brow = brow[np.argsort(bx[brow])]
    print(f"{len(t)} frames, {len(tx)} surface nodes, {len(x)} nodes in the first row (x 0 .. {x.max() / 1e3:.1f} km)")
    print("   t (yr)   Hmax (m) at x (km)   x_thick (km)  x_thin (km)   max|dZs| (m)  median|dZs| (m)  d x_thick (km)")
    prev, xprev = None, None
    for k in range(len(t)):
        zs = tz[k][row]
        zb = np.interp(x, bx[brow], bz[k][brow])
        H = zs - zb
        ih = int(np.argmax(H))
        i1 = np.where(H > a.thick)[0]
        i2 = np.where(H > a.thin)[0]
        x1 = x[i1.max()] / 1e3 if len(i1) else float("nan")
        x2 = x[i2.max()] / 1e3 if len(i2) else float("nan")
        if prev is None:
            dmax = dmed = dx = float("nan")
        else:
            dz = np.abs(tz[k] - prev)
            dmax, dmed, dx = float(dz.max()), float(np.median(dz)), x1 - xprev
        print(f"{t[k]:9.0f}   {H[ih]:8.1f} at {x[ih] / 1e3:7.1f}   {x1:11.1f}  {x2:11.1f}   {dmax:12.2f}  {dmed:15.3f}  {dx:12.2f}")
        prev, xprev = tz[k], x1


if __name__ == "__main__":
    main()
