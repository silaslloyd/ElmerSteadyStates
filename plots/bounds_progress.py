#!/usr/bin/env python3
"""The 2D bounds' spin-ups so far (jobs/flowline_bounds.cmd), from the Elmer runs' cache: every chunk the jobs have
finished is a file <work>/cache/<key>.npz (the state vector and the diagnostics at the chunk's end), so the progress
can be seen before the jobs write bound_*_spin.npz. A chunk belongs to the cold bound when its grounded bed hardly
slides (median basal speed below --cold-speed, default 1 m/yr: frozen at fcoeff x 1e6), else to the temperate one;
in each bound the chunks are in the order they were written (one chunk = 100 x 10 yr).

    python3 plots/bounds_progress.py WORKDIR REF.npz OUT.png [--since "2026-11-16 20:20"] [--cold-speed 1]

REF: a state on the same mesh (the layout of the cache vectors, the bed, the node coordinates), e.g. the bounds'
start work/flowline_rect_ext/bed30.npz. --since: only cache files written after this (local time; default the
bounds jobs' start, 2026-11-16 20:20). Panels: the surface and the base along x (every chunk, light to dark with time,
the bed in black); the thickness; the basal speed (log); the grounding line against model time.
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import find_name, load_state  # noqa: E402


def arg(args, name, default):
    if name in args:
        k = args.index(name)
        v = args[k + 1]
        del args[k:k + 2]
        return v
    return default


def main():
    args = sys.argv[1:]
    since = arg(args, "--since", "2026-11-16 20:20")
    cold_speed = float(arg(args, "--cold-speed", "1.0"))
    if len(args) != 3:
        raise SystemExit(__doc__)
    work, ref, out = args
    t_since = time.mktime(time.strptime(since, "%Y-%m-%d %H:%M"))
    s = load_state(ref)
    sl = s.layout.slices()
    blk = {b.name.lower(): b for b in s.layout.blocks}
    rec = {**s.diag, **s.frozen}
    gx, vx = rec[find_name(rec, "coordinate 1")]
    gy, vy = rec[find_name(rec, "coordinate 2")]
    X = dict(zip(gx.tolist(), vx)); Y = dict(zip(gy.tolist(), vy))
    gb, vb = s.frozen[find_name(s.frozen, "bedrock")]
    B = dict(zip(gb.tolist(), vb))
    zb_g = blk["zb"].gids
    zs_g = blk["zs"].gids
    # one node line along the flowline (the mesh is one element wide): the base nodes at the smallest y
    yb = np.array([Y[g] for g in zb_g])
    line = np.abs(yb - yb.min()) < 1.0
    bg = zb_g[line]
    bx = np.array([X[g] for g in bg])
    o = np.argsort(bx)
    bg, bx = bg[o], bx[o] / 1e3
    # the surface node above each base node: the nearest surface node in (x, y) on that line
    sxy = np.array([[X[g], Y[g]] for g in zs_g])
    from scipy.spatial import cKDTree
    js = cKDTree(sxy).query(np.array([[X[g], Y[g]] for g in bg]))[1]
    sg = zs_g[js]
    bed = np.array([B.get(g, np.nan) for g in bg])
    ib = {g: i for i, g in enumerate(zb_g)}
    isf = {g: i for i, g in enumerate(zs_g)}
    tg = blk["temperature"].gids
    it = {g: i for i, g in enumerate(tg)}

    files = [f for f in glob.glob(os.path.join(work, "cache", "*.npz")) if not f.endswith(".tmp.npz")
             and os.path.getmtime(f) > t_since]
    files.sort(key=os.path.getmtime)
    chunks = {"cold": [], "temperate": []}
    for f in files:
        with np.load(f) as z:
            x = z["x"]
            if len(x) != s.layout.size:
                continue
            diag = {k[3:]: (z["g::" + k[3:]], z[k]) for k in z.files if k.startswith("v::")}
        zb = x[sl[blk["zb"].name]][[ib[g] for g in bg]]
        zs = x[sl[blk["zs"].name]][[isf[g] for g in sg]]
        T = x[sl[blk["temperature"].name]][[it[g] for g in bg]]
        gu, vu = diag[find_name(diag, "velocity 1")]
        U = dict(zip(gu.tolist(), vu))
        ub = np.array([U.get(g, np.nan) for g in bg])
        gm = None
        try:
            gg, vg = diag[find_name(diag, "groundedmask")]
            G = dict(zip(gg.tolist(), vg))
            gm = np.array([G.get(g, np.nan) for g in bg])
        except KeyError:
            pass
        grounded = (gm > -0.5) if gm is not None else (zb - bed < 1.0)
        if not grounded.any():
            continue
        which = "cold" if np.nanmedian(np.abs(ub[grounded])) < cold_speed else "temperate"
        xg = float(bx[grounded].max())
        chunks[which].append(dict(zs=zs, zb=zb, T=T, ub=ub, xg=xg, mtime=os.path.getmtime(f)))
    for w in chunks:
        print(f"{w}: {len(chunks[w])} chunks" + (f", grounding line {chunks[w][0]['xg']:.1f} -> {chunks[w][-1]['xg']:.1f} km"
                                                 if chunks[w] else ""))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(15, 16), gridspec_kw=dict(height_ratios=[3, 1.4, 1.4, 1.4]))
    cmaps = {"cold": plt.cm.Blues, "temperate": plt.cm.Reds}
    for w, cs in chunks.items():
        n = len(cs)
        for k, c in enumerate(cs):
            col = cmaps[w](0.3 + 0.7 * (k + 1) / max(n, 1))
            last = k == n - 1
            lab = f"{w}, chunk {k + 1} ({(k + 1)} kyr)" if last else None
            ax[0].plot(bx, c["zs"], color=col, lw=1.6 if last else 0.6, label=lab)
            ax[0].plot(bx, c["zb"], color=col, lw=1.6 if last else 0.6)
            if last:
                ax[1].plot(bx, c["zs"] - c["zb"], color=col, lw=1.4, label=f"{w}, {n} kyr")
                ax[2].semilogy(bx, np.maximum(np.abs(c["ub"]), 1e-4), color=col, lw=1.4, label=f"{w}, {n} kyr")
        if n:
            ax[3].plot(np.arange(1, n + 1), [c["xg"] for c in cs], "o-", color=cmaps[w](0.8), label=w)
    ax[0].plot(bx, bed, color="k", lw=1.2, label="bed")
    ax[0].axhline(0, color="0.6", lw=0.6, ls="--")
    ax[0].set_ylabel("elevation (m)")
    ax[0].set_title("2D bounds: surface and base after every chunk (light: early, dark: latest)")
    ax[0].legend(loc="upper right", fontsize=9)
    ax[1].set_ylabel("thickness (m)")
    ax[1].set_title("ice thickness, latest chunk")
    ax[2].set_ylabel("basal speed (m/yr)")
    ax[2].set_title("basal speed along x (velocity 1), latest chunk")
    for a in ax[:3]:
        a.set_xlabel("x (km)")
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=9)
    ax[2].legend(fontsize=9)
    ax[3].set_xlabel("chunk (1000 yr each)")
    ax[3].set_ylabel("grounding line (km)")
    ax[3].set_title("grounding line against model time")
    ax[3].grid(alpha=0.3)
    ax[3].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
