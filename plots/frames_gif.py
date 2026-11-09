#!/usr/bin/env python3
"""
A top-down (map view) GIF of long runs: one picture per output step of plots/vtu_frames.py frames files, the surface
speed in the colours of plots/plot_state_3d.py (Schoof & Mantelli 2021, 0 .. 100 m/yr), with the streamlines of the
surface velocity (white), the grounding line (red) and the cold/temperate boundary of the grounded bed (magenta).

    python3 plots/frames_gif.py FRAMES.npz [FRAMES2.npz ...] [options]
        --out FILE.gif        (default: the first file's name with .gif)
        --field speed|T|vz    (default speed; T: the surface temperature, vz: the surface vertical velocity)
        --vmax V              (speed colour maximum, m/yr, default 100)
        --every N             every N-th frame (and the last)          --years T0 T1   only frames in that time range
        --x XMIN XMAX         (km, default the whole strip)            --tile N        copies across y (default 1)
        --fps F               (frames per second, default 4)           --dpi D         (default 100)
        --stream-spacing KM   (default 5)  --no-streamlines  --no-mask  --no-gl
        --compare FILE        a second run's map below the first's, at the times both have (same colours)

The files are joined in the order given (e.g. a run and its continuation). Needs matplotlib and Pillow.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_state_3d as P3  # noqa: E402  (SPEED_CS, streamlines)


def opt(args, name, n=1, conv=float, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = [conv(a) for a in args[k + 1:k + 1 + n]]
    return args[:k] + args[k + 1 + n:], (v[0] if n == 1 else v)


def flag(args, name):
    return ([a for a in args if a != name], name in args)


def main():
    args = sys.argv[1:]
    args, out = opt(args, "--out", conv=str)
    args, field = opt(args, "--field", conv=str, default="speed")
    args, vmax = opt(args, "--vmax", default=100.0)
    args, every = opt(args, "--every", conv=int, default=1)
    args, years = opt(args, "--years", n=2)
    args, xwin = opt(args, "--x", n=2)
    args, tile = opt(args, "--tile", conv=int, default=1)
    args, fps = opt(args, "--fps", default=4.0)
    args, dpi = opt(args, "--dpi", conv=int, default=100)
    args, spacing = opt(args, "--stream-spacing", default=5.0)
    args, nostream = flag(args, "--no-streamlines")
    args, nomask = flag(args, "--no-mask")
    args, nogl = flag(args, "--no-gl")
    args, compare = opt(args, "--compare", conv=str)      # --compare FILE: a second run below the first
    files = [a for a in args if a.endswith(".npz")]
    if not files or len(files) != len(args):
        raise SystemExit(__doc__)
    out = out or os.path.splitext(files[0])[0] + ".gif"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.animation import PillowWriter

    # every frame of every file, in order: (file index, frame index, time)
    D = [np.load(f, allow_pickle=True) for f in files]
    items = [(i, k, float(t)) for i, d in enumerate(D) for k, t in enumerate(d["frame_times"])]
    C, cidx = None, {}
    if compare:                                            # only the times the compared run also has (before --every)
        C = np.load(compare, allow_pickle=True)
        cidx = {round(float(t), 0): k for k, t in enumerate(C["frame_times"])}
        items = [it for it in items if round(it[2], 0) in cidx]
    if years:
        items = [it for it in items if years[0] - 1e-6 <= it[2] <= years[1] + 1e-6]
    if every > 1 and len(items) > 1:
        items = [it for n, it in enumerate(items) if n % every == 0 or n == len(items) - 1]
    if not items:
        raise SystemExit("no frames left (--years / --every / --compare)")

    if field == "speed":
        cmap = LinearSegmentedColormap.from_list("sm2021", [(v, c) for v, c in P3.SPEED_CS])
        lo, hi, label = 0.0, vmax, "surface speed (m/yr)"
    elif field == "T":
        allT = np.concatenate([d["top_T"].ravel() for d in D if "top_T" in d.files])
        cmap, lo, hi, label = "RdBu_r", float(np.nanmin(allT)), float(np.nanmax(allT)), "surface temperature (K)"
    elif field == "vz":
        allv = np.abs(np.concatenate([d["top_vz"].ravel() for d in D]))
        hi = float(np.nanpercentile(allv, 99)); cmap, lo, label = "RdBu_r", -hi, "surface vz (m/yr)"
    else:
        raise SystemExit("--field speed | T | vz")

    tris = {}
    W = max(float(np.ptp(d["top_y"])) for d in D + ([C] if C is not None else []))
    x0 = min(float(d["top_x"].min()) for d in D) / 1e3
    x1 = max(float(d["top_x"].max()) for d in D) / 1e3
    if xwin:
        x0, x1 = xwin
    Lx, Ly = x1 - x0, tile * W / 1e3
    fw = 16.0
    nrow = 2 if C is not None else 1
    mh = fw * 0.86 * Ly / Lx                          # one map's height at true aspect (inches)
    fh = max(2.5, nrow * mh + 1.0 + 0.6 * nrow)      # plus room for the titles and the x label
    fig = plt.figure(figsize=(fw, fh), dpi=dpi)
    bot, gap = 0.7 / fh, 0.6 / fh
    hh = mh / fh
    axes = [fig.add_axes([0.05, bot + (nrow - 1 - r) * (hh + gap), 0.86, hh]) for r in range(nrow)]
    cax = fig.add_axes([0.93, bot, 0.012, nrow * hh + (nrow - 1) * gap])

    def draw(ax, d, key, k, title):
        """one map: the field in colour, streamlines, grounding line, cold/temperate boundary"""
        tx, ty = d["top_x"], d["top_y"]
        if ("top", key) not in tris:
            tris[("top", key)] = mtri.Triangulation(tx / 1e3, ty / 1e3)
            tris[("base", key)] = mtri.Triangulation(d["base_x"] / 1e3, d["base_y"] / 1e3)
        vx, vy, vz = d["top_vx"][k], d["top_vy"][k], d["top_vz"][k]
        val = {"speed": lambda: np.sqrt(vx ** 2 + vy ** 2 + vz ** 2), "T": lambda: d["top_T"][k],
               "vz": lambda: vz}[field]()
        pc = None
        for c in range(tile):
            tr = mtri.Triangulation(tx / 1e3, ty / 1e3 + c * W / 1e3, tris[("top", key)].triangles)
            pc = ax.tripcolor(tr, val, shading="gouraud", cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
        if not nostream:
            top = dict(x=tx, y=ty, vx=vx, vy=vy, v=d["top_z"][k])
            P3._STREAM_CACHE.clear()
            sx, sy, _ = P3.streamlines(top, spacing, W, periodic=True)
            for c in range(tile):
                ax.plot(sx, sy + c * W / 1e3, color="white", lw=0.7)
        bt = tris[("base", key)]
        for c in range(tile):
            bc = mtri.Triangulation(bt.x, bt.y + c * W / 1e3, bt.triangles)
            if not nogl and "base_mask" in d.files:
                ax.tricontour(bc, d["base_mask"][k], levels=[0.0], colors="red", linewidths=1.6)
            if not nomask and "base_ct" in d.files and "base_mask" in d.files:
                fm = np.where(d["base_mask"][k] > -0.5, np.nan_to_num(d["base_ct"][k], nan=-1.0), -1.0)
                if fm.min() < 0 < fm.max():
                    ax.tricontour(bc, fm, levels=[0.0], colors="#ff00e0", linewidths=1.2)
        ax.set_xlim(x0, x1); ax.set_ylim(0, Ly); ax.set_aspect("equal")
        ax.set_ylabel("y (km)")
        ax.set_title(title, loc="left", fontsize=11)
        return pc

    writer = PillowWriter(fps=fps)
    print(f"{len(items)} frames -> {out} ({field}, x {x0:g} .. {x1:g} km, {tile} cop{'y' if tile == 1 else 'ies'})"
          + (f", beside {os.path.basename(compare)}" if C is not None else ""))
    name = lambda f: os.path.basename(f)[:-4].replace("_frames", "")
    with writer.saving(fig, out, dpi):
        for n, (i, k, t) in enumerate(items):
            for a in axes + [cax]:
                a.clear()
            lab = name(files[i]) if C is not None else ""
            pc = draw(axes[0], D[i], ("main", i), k, f"t = {t:,.0f} yr" + (f"   {lab}" if lab else ""))
            if C is not None:
                draw(axes[1], C, ("cmp", 0), cidx[round(t, 0)], f"t = {t:,.0f} yr   {name(compare)}")
            axes[-1].set_xlabel("x (km)")
            fig.colorbar(pc, cax=cax, label=label)
            writer.grab_frame()
            print(f"  {n + 1}/{len(items)}  t {t:g} yr", end="\r", flush=True)
    print(f"\n{out}: {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
