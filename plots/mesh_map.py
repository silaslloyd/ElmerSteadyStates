#!/usr/bin/env python3
"""A footprint mesh as a 2D map (PNG): the triangles' edges, coloured by element size (the mean edge length), with
an optional zoom. Reads an Elmer 2D mesh directory (case/<mesh_db>: mesh.nodes, mesh.elements), e.g. one made by
`ess remesh`.

    python3 plots/mesh_map.py MESHDIR OUT.png [--zoom X0 X1 Y0 Y1 (km)] [--state S.npz]
                              [--compare OTHER_MESHDIR [--compare-state S2.npz]]

    python3 plots/mesh_map.py --all WORKDIR [CASEDIR] [--redo]   (the pictures of every m### not drawn yet, to
                                                                 WORKDIR/meshes; --redo: all of them again)

The _full.png pictures show x >= X0_FULL (600 km; the whole mesh if it ends before that), true aspect, the figure
sized to the strip; triangle edges only where they are at least ~6 pixels long (else the colour alone).

--compare: a second panel with another mesh (e.g. the original footprint) on the same axes.
--state: a state on that mesh (e.g. <work>/m001_start.npz; for the original footprint the state remeshed from):
its grounding line (red: GroundedMask -0.5, between grounded and floating) and cold/temperate transition on the
grounded bed (white: ColdTempMask 0) drawn on top.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.remesh import base_values, read_footprint  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402


def curves(ax, x, y, tri, state):
    """the grounding line and the cold/temperate transition of a state on this footprint"""
    s = load_state(state)
    f = {**s.diag, **s.frozen}
    nb = len(x)
    gm = base_values(f[find_name(f, "groundedmask")], nb)
    ct = base_values(f[find_name(f, "coldtempmask")], nb)
    if np.isnan(gm).all():
        raise SystemExit(f"{state}: its GroundedMask is not on this mesh ({nb} footprint nodes)")
    gm, ct = np.nan_to_num(gm, nan=-1.0), np.nan_to_num(ct, nan=-1.0)
    ax.tricontour(x, y, tri, gm, levels=[-0.5], colors="red", linewidths=1.6, linestyles="solid")
    grounded = (gm[tri] > -0.5).all(1)
    import matplotlib.tri as mtri
    T = mtri.Triangulation(x, y, tri, mask=~grounded)
    ax.tricontour(T, ct, levels=[0.0], colors="white", linewidths=0.9, linestyles="solid")


X0_FULL = 600.0          # km: the _full pictures from here to the front


def extent(meshdir):
    _, x, y, _ = read_footprint(meshdir)
    return x.min() / 1e3, x.max() / 1e3, y.min() / 1e3, y.max() / 1e3


def full_zoom(meshdir):
    """x >= X0_FULL to the end, the whole width (the whole mesh if it ends before X0_FULL)"""
    x0, x1, y0, y1 = extent(meshdir)
    return [X0_FULL if x1 > X0_FULL + 10 else x0, x1, y0, y1]


def figure(nrows, zoom, width=16.0):
    """a figure of nrows panels of the zoom box at true aspect, sized to them (no empty space), a horizontal colour
    bar below; returns fig, axs (nrows x 1), the colour bar's axes and pixels per km at dpi 130"""
    import matplotlib.pyplot as plt
    wkm, hkm = zoom[1] - zoom[0], zoom[3] - zoom[2]
    ph = max(width * hkm / wkm, 1.0)          # inches: a panel's plot area (true aspect)
    top, below = 0.45, 0.75                   # inches: the title above a panel, the x label below it
    cbh = 1.2                                 # inches: the colour bar and its label at the bottom
    left = 0.9
    W, H = width + left + 0.3, nrows * (top + ph + below) + cbh
    fig = plt.figure(figsize=(W, H))
    axs = np.array([[fig.add_axes([left / W, (cbh + (nrows - 1 - i) * (top + ph + below) + below) / H,
                                   width / W, ph / H])] for i in range(nrows)])
    cax = fig.add_axes([(left + 0.25 * width) / W, 0.6 / H, 0.5 * width / W, 0.2 / H])
    return fig, axs, cax, width * 130.0 / wkm


def colorbar(fig, tc, cax):
    from matplotlib.ticker import FixedLocator, FuncFormatter
    cb = fig.colorbar(tc, cax=cax, orientation="horizontal")
    cb.set_label("element size (m, mean edge)")
    cb.ax.xaxis.set_major_locator(FixedLocator([250, 500, 1000, 2000, 5000]))
    cb.ax.xaxis.set_minor_locator(FixedLocator([]))
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    return cb


def panel(ax, meshdir, zoom, norm, state=None, px_per_km=None):
    from matplotlib.collections import LineCollection
    _, x, y, tri = read_footprint(meshdir)
    x, y = x / 1e3, y / 1e3
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
    e = np.unique(np.sort(e, 1), axis=0)
    L = np.hypot(x[e[:, 0]] - x[e[:, 1]], y[e[:, 0]] - y[e[:, 1]]) * 1e3
    hs = np.mean([np.hypot(x[tri[:, a]] - x[tri[:, b]], y[tri[:, a]] - y[tri[:, b]]) for a, b in
                  ((0, 1), (1, 2), (2, 0))], 0) * 1e3
    tc = ax.tripcolor(x, y, tri, facecolors=hs, cmap="viridis_r", norm=norm, edgecolors="none")
    if zoom:              # clamped to the mesh (no empty margins)
        zoom = [max(zoom[0], x.min()), min(zoom[1], x.max()), max(zoom[2], y.min()), min(zoom[3], y.max())]
    else:
        zoom = [x.min(), x.max(), y.min(), y.max()]
    m = ((x[e[:, 0]] > zoom[0]) & (x[e[:, 0]] < zoom[1]) & (y[e[:, 0]] > zoom[2]) & (y[e[:, 0]] < zoom[3]))
    ee = e[m]
    ax.set_xlim(zoom[0], zoom[1])
    ax.set_ylim(zoom[2], zoom[3])
    # edges only where they are at least ~6 pixels long at the figure's width (else they blacken the panel)
    if px_per_km:
        Le = np.hypot(x[ee[:, 0]] - x[ee[:, 1]], y[ee[:, 0]] - y[ee[:, 1]])
        ee = ee[Le * px_per_km >= 6.0]
    lw = 0.15 if len(ee) > 2e5 else 0.3
    if len(ee):
        ax.add_collection(LineCollection(np.stack([np.c_[x[ee[:, 0]], y[ee[:, 0]]],
                                                   np.c_[x[ee[:, 1]], y[ee[:, 1]]]], 1),
                                         colors="k", linewidths=lw, alpha=0.6))
    if state:
        curves(ax, x, y, tri, state)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"{os.path.basename(os.path.normpath(meshdir))}: {len(x)} nodes, {len(tri)} triangles, "
                 f"edges {L.min():.0f}..{L.max():.0f} m")
    return tc


def make_pngs(meshdir, state, outdir, name, prev_mesh=None, prev_state=None):
    """the PNGs of a new mesh (ess run --remesh-auto, ess remesh): <outdir>/<name>_full.png (beside the previous mesh
    when given) and <name>_gl.png (zoom on the grounding line, +-40 km around it); returns their paths"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    os.makedirs(outdir, exist_ok=True)
    norm = LogNorm(200, 6000)
    out = []
    meshes = [(meshdir, state)] + ([(prev_mesh, prev_state)] if prev_mesh else [])
    z = full_zoom(meshdir)
    fig, axs, cax, ppk = figure(len(meshes), z)
    for ax, (m, st) in zip(axs[:, 0], meshes):
        tc = panel(ax, m, z, norm, st, ppk)
    colorbar(fig, tc, cax)
    p = os.path.join(outdir, f"{name}_full.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    out.append(p)
    # the grounding line's extent
    _, x, y, tri = read_footprint(meshdir)
    s = load_state(state)
    f = {**s.diag, **s.frozen}
    gm = np.nan_to_num(base_values(f[find_name(f, "groundedmask")], len(x)), nan=-1.0)
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]]]
    mix = (gm[e[:, 0]] > -0.5) != (gm[e[:, 1]] > -0.5)
    if mix.any():
        gx, gy = x[e[mix].ravel()] / 1e3, y[e[mix].ravel()] / 1e3
        z = [gx.min() - 40, gx.max() + 40, gy.min() - 40, gy.max() + 40]
        x0, x1, y0, y1 = extent(meshdir)
        z = [max(z[0], x0), min(z[1], x1), max(z[2], y0), min(z[3], y1)]
        fig, axs, cax, ppk = figure(1, z, width=12.0)
        tc = panel(axs[0, 0], meshdir, z, norm, state, ppk)
        colorbar(fig, tc, cax)
        p = os.path.join(outdir, f"{name}_gl.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


def make_all(work, case, redo=False):
    """the pictures of every remeshed mesh (case/[<label>_]m###, its state <work>/<name>_start.npz) that has none yet in
    <work>/meshes (redo: all of them)"""
    import glob
    made = []
    for st in sorted(glob.glob(os.path.join(work, "*m[0-9][0-9][0-9]_start.npz"))):
        name = os.path.basename(st)[:-len("_start.npz")]
        if (os.path.exists(os.path.join(work, "meshes", f"{name}_full.png")) and not redo) \
                or not os.path.isdir(os.path.join(case, name)):
            continue
        made += make_pngs(os.path.join(case, name), st, os.path.join(work, "meshes"), name)
    print("made: " + (" ".join(made) if made else "nothing new"))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--all":       # mesh_map.py --all WORK [CASE]: every new mesh's pictures
        rest = [v for v in sys.argv[2:] if v != "--redo"]
        w = rest[0]
        make_all(w, rest[1] if len(rest) > 1 else os.path.join(w, "case"), redo="--redo" in sys.argv)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--zoom", type=float, nargs=4)
    ap.add_argument("--compare")
    ap.add_argument("--state")
    ap.add_argument("--compare-state")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    norm = LogNorm(200, 6000)
    meshes = [a.mesh] + ([a.compare] if a.compare else [])
    x0, x1, y0, y1 = extent(a.mesh)
    z = a.zoom or [x0, x1, y0, y1]
    z = [max(z[0], x0), min(z[1], x1), max(z[2], y0), min(z[3], y1)]
    fig, axs, cax, ppk = figure(len(meshes), z, width=16.0 if not a.zoom else 12.0)
    states = [a.state] + ([a.compare_state] if a.compare else [])
    for ax, m, st in zip(axs[:, 0], meshes, states):
        tc = panel(ax, m, z, norm, st, ppk * 150 / 130)
    colorbar(fig, tc, cax)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
