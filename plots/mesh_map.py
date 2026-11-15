#!/usr/bin/env python3
"""A footprint mesh as a 2D map (PNG): the triangles' edges, coloured by element size (the mean edge length), with
an optional zoom. Reads an Elmer 2D mesh directory (case/<mesh_db>: mesh.nodes, mesh.elements), e.g. one made by
`ess remesh`.

    python3 plots/mesh_map.py MESHDIR OUT.png [--zoom X0 X1 Y0 Y1 (km)] [--compare OTHER_MESHDIR]

--compare: a second panel with another mesh (e.g. the original footprint) on the same axes.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.remesh import read_footprint  # noqa: E402


def panel(ax, meshdir, zoom, norm):
    from matplotlib.collections import LineCollection
    _, x, y, tri = read_footprint(meshdir)
    x, y = x / 1e3, y / 1e3
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
    e = np.unique(np.sort(e, 1), axis=0)
    L = np.hypot(x[e[:, 0]] - x[e[:, 1]], y[e[:, 0]] - y[e[:, 1]]) * 1e3
    hs = np.mean([np.hypot(x[tri[:, a]] - x[tri[:, b]], y[tri[:, a]] - y[tri[:, b]]) for a, b in
                  ((0, 1), (1, 2), (2, 0))], 0) * 1e3
    tc = ax.tripcolor(x, y, tri, facecolors=hs, cmap="viridis_r", norm=norm, edgecolors="none")
    if zoom:
        m = ((x[e[:, 0]] > zoom[0]) & (x[e[:, 0]] < zoom[1]) & (y[e[:, 0]] > zoom[2]) & (y[e[:, 0]] < zoom[3]))
        ee = e[m]
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    else:
        ee = e
    lw = 0.15 if len(ee) > 2e5 else 0.3
    ax.add_collection(LineCollection(np.stack([np.c_[x[ee[:, 0]], y[ee[:, 0]]], np.c_[x[ee[:, 1]], y[ee[:, 1]]]], 1),
                                     colors="k", linewidths=lw, alpha=0.6))
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"{os.path.basename(os.path.normpath(meshdir))}: {len(x)} nodes, {len(tri)} triangles, "
                 f"edges {L.min():.0f}..{L.max():.0f} m")
    return tc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--zoom", type=float, nargs=4)
    ap.add_argument("--compare")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    norm = LogNorm(400, 6000)
    meshes = [a.mesh] + ([a.compare] if a.compare else [])
    fig, axs = plt.subplots(len(meshes), 1, figsize=(16, 6 * len(meshes)) if not a.zoom else (12, 10 * len(meshes)),
                            squeeze=False)
    for ax, m in zip(axs[:, 0], meshes):
        tc = panel(ax, m, a.zoom, norm)
    fig.colorbar(tc, ax=axs[:, 0].tolist(), label="element size (m, mean edge)", shrink=0.8)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
