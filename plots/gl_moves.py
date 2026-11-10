#!/usr/bin/env python3
"""Where the grounding line moves between frames (plots/vtu_frames.py file): the grounding-line nodes (grounded nodes
of triangles with a floating node) of each frame against the previous frame's, the largest distance either way (the
Hausdorff distance GLDistance's "FS SSA Update Distance" compares with) and WHERE, and the median shift. A grounding
line that is still but for a few flickering nodes (an island, a seam node) shows a large maximum and a ~0 median.

    python3 plots/gl_moves.py FRAMES.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_state_3d import triangles  # noqa: E402

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
F = np.load(sys.argv[1])
x, y = F["base_x"].astype(float), F["base_y"].astype(float)
i, j, k = triangles(x, y)
T = np.c_[i, j, k]
from scipy.spatial import cKDTree  # noqa: E402


def glnodes(m):
    g = m[T] > -0.5
    mixed = g.any(1) & (~g).any(1)
    nodes = np.unique(T[mixed][g[mixed]])
    return nodes


prev = None
for n, t in enumerate(F["frame_times"]):
    G = glnodes(F["base_mask"][n])
    P = np.c_[x[G], y[G]]
    if prev is not None:
        d1, _ = cKDTree(prev).query(P)
        d0, _ = cKDTree(P).query(prev)
        a, b = np.argmax(d1), np.argmax(d0)
        far = (P[a], d1[a]) if d1[a] >= d0[b] else (prev[b], d0[b])
        print(f"t {t:7.0f}: {len(G)} GL nodes; largest move {far[1] / 1e3:6.2f} km at x {far[0][0] / 1e3:6.1f} "
              f"y {far[0][1] / 1e3:6.1f} km; median {np.median(d1) / 1e3:5.2f} km; moved > 1 km: {int((d1 > 1e3).sum())} "
              f"new, {int((d0 > 1e3).sum())} gone")
    prev = P
