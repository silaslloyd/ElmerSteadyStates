#!/usr/bin/env python3
"""
How far each run's surface is from a reference run (e.g. FSSA full at dt 50), at the frame times they share: the rms
and the maximum of |Zs - Zs_ref| (m) and where the maximum is (x, y).

    python3 plots/fssa_compare.py REF.npz RUN.npz [RUN2.npz ...]     (frames files of plots/vtu_frames.py; 'bash plots/fssa_pack.sh')
    python3 plots/fssa_compare.py work/land_flowline_3d/cmp_full_dt50_frames.npz work/land_flowline_3d/cmp_*_frames.npz

A run is compared at the times (to 1 yr) the reference also has; nodes are matched by (x, y). Also printed: the
reference's own change over the same period (rms |Zs(t) - Zs(first common time)|), the size the differences are measured against.
"""
import sys

import numpy as np
from scipy.spatial import cKDTree


def load(p):
    z = np.load(p)
    return np.asarray(z["frame_times"], float), np.asarray(z["top_x"], float), np.asarray(z["top_y"], float), np.asarray(z["top_z"], float)


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    tr, xr, yr, zr = load(a[0])
    print(f"reference {a[0]}: {len(tr)} frames, t {tr[0]:g} .. {tr[-1]:g} yr")
    for p in a[1:]:
        if p == a[0]:
            continue
        t, x, y, z = load(p)
        d, idx = cKDTree(np.column_stack([xr, yr])).query(np.column_stack([x, y]))
        if d.max() > 5 or len(np.unique(idx)) != len(idx):
            print(f"{p}: nodes do not match the reference's (other mesh?), skipped")
            continue
        order = np.empty(len(idx), int)
        order[idx] = np.arange(len(idx))               # run node i matches reference node idx[i]
        zz = z[:, order]                               # run's Zs in the reference's node order
        rows = []
        for k, tk in enumerate(t):
            j = int(np.argmin(np.abs(tr - tk)))
            if abs(tr[j] - tk) > 1.0:
                continue
            diff = zz[k] - zr[j]
            m = int(np.argmax(np.abs(diff)))
            rows.append((tk, float(np.sqrt(np.mean(diff ** 2))), float(abs(diff[m])), xr[m] / 1e3, yr[m] / 1e3,
                         float(np.sqrt(np.mean((zr[j] - zr[0]) ** 2)))))
        print(f"\n{p}: {len(t)} frames, {len(rows)} at reference times")
        if not rows:
            continue
        print("   t (yr)   rms diff (m)   max diff (m)   at x (km)  y (km)   reference change since its first frame, rms (m)")
        for r in rows:
            print(f"{r[0]:9g}   {r[1]:11.3f}   {r[2]:11.3f}   {r[3]:8.1f} {r[4]:6.1f}   {r[5]:8.3f}")


if __name__ == "__main__":
    main()
