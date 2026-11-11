#!/usr/bin/env python3
"""A sector run tiled into a full circle: map view (PNG) of one frame of a plots/vtu_frames.py file, the sector and its
copies each rotated by the sector angle about the centre, coloured by surface speed (the Schoof & Mantelli colours of
plots/plot_state_3d.py) or temperature, with the grounding line in red. --gif: every frame, as an animated GIF.

    python3 plots/sector_circle.py FRAMES.npz [--frame K] [--field speed|T|Tbase] [--vmax V] [--copies N]
                                   [--out FILE] [--dpi D] [--gif] [--every-yr Y] [--fps F]

--frame: default the last; --copies: default as many whole sectors as fit in 360 deg (a sector angle that does not
divide 360 deg leaves a gap, marked); --vmax: speed colour maximum (default 100 m/yr).
--gif: all frames (or the first and then every Y yr with --every-yr Y, and the last) into FRAMES_circle.gif, F frames
per second (default 2), at --dpi (default 100 for a GIF, 200 for a PNG); the colour scale fixed over the frames.
On a machine where numpy's OpenBLAS dies with "Illegal instruction": OPENBLAS_CORETYPE=Nehalem.
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_state_3d as P3  # noqa: E402  (SPEED_CS, triangles, sector_angle)


def opt(args, name, conv=str, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = conv(args[k + 1])
    return args[:k] + args[k + 2:], v


def main():
    args = sys.argv[1:]
    gif = "--gif" in args
    args = [a for a in args if a != "--gif"]
    args, k = opt(args, "--frame", int)
    args, field = opt(args, "--field", str, "speed")
    args, vmax = opt(args, "--vmax", float, 100.0)
    args, ncp = opt(args, "--copies", int)
    args, out = opt(args, "--out")
    args, dpi = opt(args, "--dpi", int, 100 if gif else 200)
    args, every_yr = opt(args, "--every-yr", float)
    args, fps = opt(args, "--fps", float, 2.0)
    if len(args) != 1:
        raise SystemExit(__doc__)
    F = np.load(args[0])
    times = F["frame_times"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import LinearSegmentedColormap

    tx, ty = F["top_x"].astype(float), F["top_y"].astype(float)
    bx, by = F["base_x"].astype(float), F["base_y"].astype(float)
    ph = np.arctan2(ty, tx)
    A = P3.sector_angle(tx, ty)                         # (the outer arc's span: also for jagged cut sides)
    if ncp is None:
        ncp = int(np.floor(2 * np.pi / A + 1e-6))
    gap = 2 * np.pi - ncp * A
    if field == "speed":
        cmap = LinearSegmentedColormap.from_list("sm2021", [(v, c) for v, c in P3.SPEED_CS])
        lab = "surface speed (m/yr)"
        X, Y = tx, ty
        value = lambda kk: np.sqrt(F["top_vx"][kk].astype(float) ** 2 + F["top_vy"][kk].astype(float) ** 2)
        lo, hi = 0.0, vmax
    elif field in ("T", "Tbase"):
        key = "top_T" if field == "T" else "base_T"
        cmap = "RdBu_r"
        lab = "surface temperature (K)" if field == "T" else "basal temperature (K)"
        X, Y = (tx, ty) if field == "T" else (bx, by)
        value = lambda kk: F[key][kk].astype(float)
        sel = range(len(times)) if gif else [len(times) - 1 if k is None else k]
        lo = float(min(np.nanmin(value(kk)) for kk in sel))     # one scale over all frames drawn
        hi = float(max(np.nanmax(value(kk)) for kk in sel))
    else:
        raise SystemExit("--field speed | T | Tbase")
    i, j, kk_ = P3.triangles(X, Y)
    bi, bj, bk = P3.triangles(bx, by)
    rmax = max(np.hypot(tx, ty).max(), np.hypot(bx, by).max()) / 1e3 * 1.03
    # the rotated copies' triangulations once (the footprint does not change)
    tri_v, tri_b = [], []
    for c in range(ncp):
        ca, sa = np.cos(c * A), np.sin(c * A)
        tri_v.append(mtri.Triangulation((ca * X - sa * Y) / 1e3, (sa * X + ca * Y) / 1e3, np.c_[i, j, kk_]))
        tri_b.append(mtri.Triangulation((ca * bx - sa * by) / 1e3, (sa * bx + ca * by) / 1e3, np.c_[bi, bj, bk]))

    def draw(kk):
        val = value(kk)
        mask = F["base_mask"][kk].astype(float) if "base_mask" in F.files else None
        fig, ax = plt.subplots(figsize=(10, 10))
        pc = None
        for c in range(ncp):
            pc = ax.tripcolor(tri_v[c], val, shading="gouraud", cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
            if mask is not None:
                ax.tricontour(tri_b[c], mask, levels=[-0.5], colors="red", linewidths=0.8)
        if gap > 1e-3:                                 # the part of the circle no whole sector fills
            r1 = np.hypot(tx, ty).max() / 1e3
            a0, a1 = -A / 2 + ncp * A, -A / 2 + 2 * np.pi
            ang = np.linspace(a0, a1, 50)
            ax.fill(np.r_[0, r1 * 1.02 * np.cos(ang), 0], np.r_[0, r1 * 1.02 * np.sin(ang), 0], color="0.85", zorder=0)
        ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
        ax.set_title(f"{os.path.basename(args[0])[:-4]}: t = {times[kk]:,.0f} yr, {ncp} copies of the "
                     f"{np.degrees(A):.3f} deg sector" + (f" (gap {np.degrees(gap):.1f} deg)" if gap > 1e-3 else ""),
                     fontsize=10)
        fig.colorbar(pc, ax=ax, shrink=0.7, label=lab)
        return fig

    if not gif:
        kk = len(times) - 1 if k is None else k
        fig = draw(kk)
        out = out or os.path.splitext(args[0])[0] + f"_circle_t{times[kk]:.0f}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"{out}: {ncp} copies, t = {times[kk]:g} yr")
        return

    from PIL import Image
    sel = list(range(len(times)))
    if every_yr:
        sel = [n for n in sel if n in (0, len(times) - 1)
               or abs(times[n] / every_yr - round(times[n] / every_yr)) < 1e-6]
    imgs = []
    for n in sel:
        fig = draw(n)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, format="png")      # (no bbox_inches="tight": every frame the same size)
        plt.close(fig)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT
                                                             if hasattr(Image, "Quantize") else 0))
        print(f"  frame {len(imgs)}/{len(sel)}: t = {times[n]:g} yr", flush=True)
    out = out or os.path.splitext(args[0])[0] + "_circle.gif"
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=False)
    print(f"{out}: {len(imgs)} frames, {ncp} copies, t {times[sel[0]]:g} .. {times[sel[-1]]:g} yr "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
