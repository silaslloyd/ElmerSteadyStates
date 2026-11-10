#!/usr/bin/env python3
"""A sector run tiled into a full circle: map view (PNG) of one frame of a plots/vtu_frames.py file, the sector and its
copies each rotated by the sector angle about the centre, coloured by surface speed (the Schoof & Mantelli colours of
plots/plot_state_3d.py) or temperature, with the grounding line in red.

    python3 plots/sector_circle.py FRAMES.npz [--frame K] [--field speed|T|Tbase] [--vmax V] [--copies N]
                                   [--out FILE.png] [--dpi D]

--frame: default the last; --copies: default as many whole sectors as fit in 360 deg (a sector angle that does not
divide 360 deg leaves a gap, marked); --vmax: speed colour maximum (default 100 m/yr).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_state_3d as P3  # noqa: E402  (SPEED_CS, triangles)


def opt(args, name, conv=str, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = conv(args[k + 1])
    return args[:k] + args[k + 2:], v


def main():
    args = sys.argv[1:]
    args, k = opt(args, "--frame", int)
    args, field = opt(args, "--field", str, "speed")
    args, vmax = opt(args, "--vmax", float, 100.0)
    args, ncp = opt(args, "--copies", int)
    args, out = opt(args, "--out")
    args, dpi = opt(args, "--dpi", int, 200)
    if len(args) != 1:
        raise SystemExit(__doc__)
    F = np.load(args[0])
    times = F["frame_times"]
    k = len(times) - 1 if k is None else k
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import LinearSegmentedColormap

    tx, ty = F["top_x"].astype(float), F["top_y"].astype(float)
    bx, by = F["base_x"].astype(float), F["base_y"].astype(float)
    ph = np.arctan2(ty, tx)
    A = float(ph.max() - ph.min())
    if ncp is None:
        ncp = int(np.floor(2 * np.pi / A + 1e-6))
    gap = 2 * np.pi - ncp * A
    if field == "speed":
        val = np.sqrt(F["top_vx"][k].astype(float) ** 2 + F["top_vy"][k].astype(float) ** 2)
        cmap = LinearSegmentedColormap.from_list("sm2021", [(v, c) for v, c in P3.SPEED_CS])
        lo, hi, lab = 0.0, vmax, "surface speed (m/yr)"
        X, Y = tx, ty
    elif field in ("T", "Tbase"):
        key = "top_T" if field == "T" else "base_T"
        val = F[key][k].astype(float)
        cmap, lo, hi = "RdBu_r", float(np.nanmin(val)), float(np.nanmax(val))
        lab = "surface temperature (K)" if field == "T" else "basal temperature (K)"
        X, Y = (tx, ty) if field == "T" else (bx, by)
    else:
        raise SystemExit("--field speed | T | Tbase")
    i, j, kk = P3.triangles(X, Y)
    bi, bj, bk = P3.triangles(bx, by)
    mask = F["base_mask"][k].astype(float) if "base_mask" in F.files else None

    fig, ax = plt.subplots(figsize=(10, 10))
    pc = None
    for c in range(ncp):
        a = c * A
        ca, sa = np.cos(a), np.sin(a)
        tr = mtri.Triangulation((ca * X - sa * Y) / 1e3, (sa * X + ca * Y) / 1e3, np.c_[i, j, kk])
        pc = ax.tripcolor(tr, val, shading="gouraud", cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
        if mask is not None:
            trb = mtri.Triangulation((ca * bx - sa * by) / 1e3, (sa * bx + ca * by) / 1e3, np.c_[bi, bj, bk])
            ax.tricontour(trb, mask, levels=[-0.5], colors="red", linewidths=0.8)
    if gap > 1e-3:                                     # the part of the circle no whole sector fills
        r1 = np.hypot(tx, ty).max() / 1e3
        a0, a1 = ph.min() + ncp * A, ph.min() + 2 * np.pi
        ang = np.linspace(a0, a1, 50)
        ax.fill(np.r_[0, r1 * 1.02 * np.cos(ang), 0], np.r_[0, r1 * 1.02 * np.sin(ang), 0], color="0.85", zorder=0)
        ax.text(0.7 * r1 * np.cos(0.5 * (a0 + a1)), 0.7 * r1 * np.sin(0.5 * (a0 + a1)),
                f"gap {np.degrees(gap):.1f} deg", ha="center", va="center", fontsize=8, color="0.3")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title(f"{os.path.basename(args[0])[:-4]}: t = {times[k]:,.0f} yr, {ncp} copies of the "
                 f"{np.degrees(A):.3f} deg sector" + (f" (gap {np.degrees(gap):.1f} deg)" if gap > 1e-3 else ""),
                 fontsize=10)
    fig.colorbar(pc, ax=ax, shrink=0.7, label=lab)
    out = out or os.path.splitext(args[0])[0] + f"_circle_t{times[k]:.0f}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"{out}: {ncp} copies, t = {times[k]:g} yr")


if __name__ == "__main__":
    main()
