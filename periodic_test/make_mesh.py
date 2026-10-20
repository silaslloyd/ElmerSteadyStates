#!/usr/bin/env python3
"""
Structured 2D footprint [0, L] x [0, W] of nx x ny quads, written directly in Elmer mesh format.
ElmerSolver extrudes it into the 3D slab ("Extruded Mesh Levels"), as in the ElmerStreams runs.

Boundary tags:  1 x = 0,  2 x = L,  3 y = 0,  4 y = W (periodic pairs 2 -> 1 and 4 -> 3).
After extrusion Elmer adds 5 = bottom (bed) and 6 = top (surface).
"""
import os
import sys


def write_footprint(out_dir, L, W, nx, ny):
    os.makedirs(out_dir, exist_ok=True)
    nid = lambda i, j: j * (nx + 1) + i + 1           # node (i, j), 1-based
    nodes = [(nid(i, j), L * i / nx, W * j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            elems.append((len(elems) + 1, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))
    eid = lambda i, j: j * nx + i + 1
    bnd = []
    for j in range(ny):                                # x = 0 and x = L
        bnd.append((1, eid(0, j), nid(0, j + 1), nid(0, j)))
        bnd.append((2, eid(nx - 1, j), nid(nx, j), nid(nx, j + 1)))
    for i in range(nx):                                # y = 0 and y = W
        bnd.append((3, eid(i, 0), nid(i, 0), nid(i + 1, 0)))
        bnd.append((4, eid(i, ny - 1), nid(i + 1, ny), nid(i, ny)))
    with open(os.path.join(out_dir, "mesh.header"), "w") as f:
        f.write(f"{len(nodes)} {len(elems)} {len(bnd)}\n2\n404 {len(elems)}\n202 {len(bnd)}\n")
    with open(os.path.join(out_dir, "mesh.nodes"), "w") as f:
        f.writelines(f"{n} -1 {x:.6f} {y:.6f} 0.0\n" for n, x, y in nodes)
    with open(os.path.join(out_dir, "mesh.elements"), "w") as f:
        f.writelines(f"{e} 1 404 {a} {b} {c} {d}\n" for e, a, b, c, d in elems)
    with open(os.path.join(out_dir, "mesh.boundary"), "w") as f:
        f.writelines(f"{k + 1} {t} {p} 0 202 {a} {b}\n" for k, (t, p, a, b) in enumerate(bnd))


if __name__ == "__main__":
    a = sys.argv[1:]
    write_footprint(a[0] if a else "footprint", 10e3, 10e3, 20, 20)
