#!/usr/bin/env python3
"""Check a 2D Elmer footprint mesh (mesh.nodes / mesh.elements of an ElmerGrid mesh directory): its mirror symmetry
about y = (ymin + ymax) / 2 (every node has a mirror node, every element a mirror element) and the element shapes
(aspect ratio: longest edge / height on it; 1.15 equilateral, much larger elongated), binned along x.

    python3 periodic_test/check_mesh.py work/test_periodic_gl_50km_tri/case/footprint [--bins 12]"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("mesh")
ap.add_argument("--bins", type=int, default=12)
a = ap.parse_args()
import glob
if os.path.exists(os.path.join(a.mesh, "mesh.nodes")):
    nfiles, efiles = [os.path.join(a.mesh, "mesh.nodes")], [os.path.join(a.mesh, "mesh.elements")]
else:                                   # only partitioned copies: the first partitioning, merged (global numbers)
    pd = sorted(glob.glob(os.path.join(a.mesh, "partitioning.*")))[0]
    nfiles = sorted(glob.glob(os.path.join(pd, "part.*.nodes")))
    efiles = [f.replace(".nodes", ".elements") for f in nfiles]
    print(f"(no mesh.nodes: merged from {pd})")
xy = {}
for f in nfiles:
    for line in open(f):
        s = line.split()
        xy[int(s[0])] = (float(s[2]), float(s[3]))
ids = np.array(sorted(xy))
X = np.array([xy[i][0] for i in ids]); Y = np.array([xy[i][1] for i in ids])
pos = {i: k for k, i in enumerate(ids)}
els = {}
for f in efiles:
    for line in open(f):
        s = line.split()
        if s[2] == "303":
            els[int(s[0])] = [pos[int(v)] for v in s[3:6]]
T = np.array(list(els.values()))
y0, y1 = Y.min(), Y.max()
key = lambda x, y: (round(x, 1), round(y, 1))
nodes = {key(x, y) for x, y in zip(X, Y)}
miss = sum(key(x, y0 + y1 - y) not in nodes for x, y in zip(X, Y))
ek = lambda t: frozenset(key(X[i], Y[i]) for i in t)
elset = {ek(t) for t in T}
emiss = sum(frozenset(key(X[i], y0 + y1 - Y[i]) for i in t) not in elset for t in T)
print(f"{len(X)} nodes, {len(T)} triangles, y {y0:.1f} .. {y1:.1f}")
print(f"mirror about y = {(y0 + y1) / 2:.1f}: nodes without a mirror node {miss}, triangles without a mirror triangle "
      f"{emiss}")
P = np.stack([np.c_[X[T[:, k]], Y[T[:, k]]] for k in range(3)], 1)
e = np.stack([np.linalg.norm(P[:, (k + 1) % 3] - P[:, k], axis=1) for k in range(3)], 1)
u_, v_ = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
area = 0.5 * np.abs(u_[:, 0] * v_[:, 1] - u_[:, 1] * v_[:, 0])
L = e.max(1)
ar = L / (2 * area / L)
xc = P[:, :, 0].mean(1) / 1e3
edges = np.linspace(xc.min(), xc.max(), a.bins + 1)
print(" x (km)          triangles   size (m) median   aspect median   p95    max")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (xc >= lo) & (xc <= hi)
    if m.any():
        print(f" {lo:6.0f} .. {hi:6.0f}  {m.sum():9d}   {np.median(L[m]):9.0f}        {np.median(ar[m]):6.2f}"
              f"      {np.percentile(ar[m], 95):5.2f}  {ar[m].max():6.2f}")
