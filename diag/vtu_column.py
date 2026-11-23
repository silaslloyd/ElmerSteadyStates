#!/usr/bin/env python3
"""Probe the node layout of an Elmer .vtu output: find the grounding line from `groundedmask`, then print every node of
the column at the grounding line and of one on the shelf (z, the geometry fields, mask, velocity).

    python3 diag/vtu_column.py /DATA/glaciology/ChristianFixedFine 0369 [x_km_shelf_offset]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vtu_read import read_many  # noqa: E402

folder, tag = sys.argv[1], sys.argv[2]
off = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
fs = sorted(glob.glob(os.path.join(folder, f"*np*_t{tag}*.vtu")))
P, pt = read_many(fs)
x = np.round(P[:, 0], 1)
y0 = P[:, 1].min()
on = np.isclose(P[:, 1], y0)
xs = np.unique(x[on])
gm = np.array([np.median(pt["groundedmask"][on & (x == xi)]) for xi in xs])
print("groundedmask values over the columns:", np.unique(np.round(gm, 2)))
grounded = xs[gm > -0.5]
xgl = grounded.max()
print("last grounded column x = %.2f km  (columns: %d grounded, %d floating)" % (xgl / 1e3, len(grounded), (gm <= -0.5).sum()))
cols = {"vel": pt["velocity"]}
for xc, label in ((xgl, "GROUNDING LINE"), (xs[np.argmin(np.abs(xs - (xgl + off * 1e3)))], f"{off:g} km SEAWARD")):
    m = on & (x == xc)
    o = np.argsort(P[m, 2])
    print("\n=== %s: x = %.2f km, %d nodes (y = %g)" % (label, xc / 1e3, m.sum(), y0))
    print("%9s %9s %9s %9s %6s %6s %9s %9s %9s %8s %8s" % ("z", "zs", "zb", "bedrock", "gmask", "ctmask", "height", "depth", "u_x", "u_z", "T"))
    for i in np.where(m)[0][o]:
        print("%9.2f %9.2f %9.2f %9.2f %6.2f %6.2f %9.1f %9.1f %9.3f %8.3f %8.2f" % (
            P[i, 2], pt["zs"][i], pt["zb"][i], pt["bedrock"][i], pt["groundedmask"][i], pt["coldtempmask"][i],
            pt["height"][i], pt["depth"][i], pt["velocity"][i, 0], pt["velocity"][i, 2], pt["temperature"][i]))
