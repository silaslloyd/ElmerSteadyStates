#!/usr/bin/env python3
"""What is in a (partitioned) Elmer .vtu output: the mesh extent, the point fields, and the column spacing along x.

    python3 diag/inspect_vtu.py /DATA/glaciology/ChristianFixedFine 0369      (folder, time-step tag)
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vtu_read import read_many  # noqa: E402

folder, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
fs = sorted(glob.glob(os.path.join(folder, f"*np*_t{tag}*.vtu"))) or sorted(glob.glob(os.path.join(folder, f"*t{tag}*.vtu")))
if not fs:
    raise SystemExit(f"no .vtu for tag {tag!r} in {folder}")
print(len(fs), "partition files, e.g.", os.path.basename(fs[0]))
P, pt = read_many(fs)
print("points %d   x %.2f..%.2f km   y %.1f..%.1f m   z %.1f..%.1f m" % (
    len(P), P[:, 0].min() / 1e3, P[:, 0].max() / 1e3, P[:, 1].min(), P[:, 1].max(), P[:, 2].min(), P[:, 2].max()))
print("2D (flowline) mesh" if np.ptp(P[:, 1]) < 1e-6 * max(1.0, np.ptp(P[:, 0])) or np.ptp(P[:, 2]) < 1e-9
      else "mesh has extent in y and z (check which axis is vertical)")
print("point fields:")
for k, v in pt.items():
    print("   %-40s %s   range %.4g .. %.4g" % (k, v.shape[1:] or "scalar", np.nanmin(v), np.nanmax(v)))
x = np.unique(np.round(P[:, 0], 3))
d = np.diff(x)
print("distinct x positions:", len(x))
for lo, hi in ((0, 600e3), (600e3, 700e3), (700e3, 740e3), (740e3, 800e3), (800e3, 1e9)):
    m = (x[:-1] >= lo) & (x[:-1] < hi)
    if m.any():
        print("column spacing %4.0f-%4.0f km: median %5.0f m  (min %.0f, max %.0f)" % (lo / 1e3, min(hi, x.max()) / 1e3,
              np.median(d[m]), d[m].min(), d[m].max()))
