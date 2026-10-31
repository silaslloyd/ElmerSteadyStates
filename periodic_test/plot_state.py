#!/usr/bin/env python3
"""
Map view of a marine 3D state (any footprint mesh: structured or the unstructured triangles): surface speed,
thickness, ice base, grounded mask with the grounding line, basal speed. Two steps, so only a small file travels:

1. on tethys (needs the repository, numpy only): extract the footprint fields of one or more states
    python3 periodic_test/plot_state.py extract work/test_periodic_gl_50km_tri/scaling_768_tri_W.npz \\
        [work/test_periodic_gl_50km_tri/scaling_start_avg.npz ...] --out periodic_tri_W.fp.npz
2. anywhere with numpy + matplotlib (the laptop, or tethys if it has matplotlib): plot it
    scp tethys:/SCRATCH/.../ElmerSteadyStates/periodic_tri_W.fp.npz .
    python periodic_test/plot_state.py plot periodic_tri_W.fp.npz [--xlim 650 800] [--png out.png]      (2D maps)
    python periodic_test/plot_state.py plot3d periodic_tri_W.fp.npz [--xlim 650 800] [--out state3d.html]
        (3D, interactive, any browser: surface, base and a slice along y = width/2 coloured by speed, surface flow
        arrows, the grounding line black, the full-Stokes | SSA boundary red: a line on the surface and the base and
        a curtain through the ice; --fs-ssa 10 km, --walls for walled sides, --exag 20)

Panels (x along the flow, km; y across): surface speed (log), ice thickness, ice base Zb, basal speed (log),
GroundedMask, and the grounding line (the GroundedMask = 0 contour) of every extracted state on top of each
other (the first state solid black, the others dashed), with its x range per state printed.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ---- settings for pressing Run (plot step)
FILE = "periodic_tri_W.fp.npz"
XLIM = (650.0, 800.0)          # km, None for the whole domain
PNG = None                     # e.g. "state.png"; None: a window


def extract(paths, out):
    sys.path.insert(0, REPO)
    from ess.statefile import find_name, load_state
    res = {"names": np.array([os.path.basename(p) for p in paths])}
    for k, p in enumerate(paths):
        s = load_state(p)
        allf = {**s.diag, **s.frozen}
        cx, X = allf[find_name(allf, "coordinate 1")]
        cy, Y = allf[find_name(allf, "coordinate 2")]
        xd = dict(zip(cx.tolist(), X.tolist()))
        yd = dict(zip(cy.tolist(), Y.tolist()))
        blocks = s.layout.split(s.x)
        names = {b.name.lower(): b for b in s.layout.blocks}
        bzs, bzb = names["zs"], names["zb"]
        top = bzs.gids                          # surface nodes
        base = bzb.gids                         # ice-base (interface) nodes
        key = lambda g: (round(xd[g], 1), round(yd[g], 1))
        bi = {key(g): i for i, g in enumerate(base.tolist())}
        # columns: the base nodes, with the top node of the same (x, y)
        ti = np.full(len(base), -1)
        for j, g in enumerate(top.tolist()):
            i = bi.get(key(g))
            if i is not None:
                ti[i] = j
        ok = ti >= 0
        x = np.array([xd[g] for g in base.tolist()])
        y = np.array([yd[g] for g in base.tolist()])
        zs = np.asarray(blocks[bzs.name], float)[ti]
        zb = np.asarray(blocks[bzb.name], float)

        def nodal(name, gids):
            if not any(n.lower() == name.lower() for n in allf):
                return np.full(len(gids), np.nan)
            g, v = allf[find_name(allf, name)]
            d = dict(zip(g.tolist(), np.asarray(v, float).tolist()))
            return np.array([d.get(q, np.nan) for q in gids.tolist()])
        gm = nodal("GroundedMask", base)
        bed = nodal("bedrock", base)
        us = np.hypot(nodal("Velocity 1", top), nodal("Velocity 2", top))[ti]
        ub = np.hypot(nodal("Velocity 1", base), nodal("Velocity 2", base))
        ux, uy, uz = (nodal(f"Velocity {c}", top)[ti] for c in (1, 2, 3))
        bx, by, bz = (nodal(f"Velocity {c}", base) for c in (1, 2, 3))
        gld = nodal("GLDistance", base)
        for n, v in dict(x=x, y=y, zs=zs, zb=zb, gm=gm, bed=bed, us=us, ub=ub, sux=ux, suy=uy, suz=uz,
                         bux=bx, buy=by, buz=bz, gld=gld).items():
            res[f"{n}{k}"] = v[ok]
        # a vertical slice along y = width/2 (the mirror line of the triangle mesh, a row of the structured one)
        gv, _ = allf[find_name(allf, "Velocity 1")]
        yv = np.array([yd.get(g, np.nan) for g in gv.tolist()])
        yrow = 0.5 * (np.nanmin(y) + np.nanmax(y))    # a node row on both meshes (100 elements across; the mirror line)
        m = np.abs(yv - yrow) < 1.0
        gs = gv[m]
        cz = allf[find_name(allf, "coordinate 3")]
        zd = dict(zip(cz[0].tolist(), np.asarray(cz[1], float).tolist()))
        res[f"slx{k}"] = np.array([xd[g] for g in gs.tolist()])
        res[f"slz{k}"] = np.array([zd[g] for g in gs.tolist()])
        res[f"slu{k}"] = np.stack([nodal(f"Velocity {c}", gs) for c in (1, 2, 3)], axis=1)
        res[f"sly{k}"] = np.array([yrow])
        print(f"{p}: {ok.sum()} columns, grounded {np.sum(gm[ok] > -0.5)}, largest surface speed "
              f"{np.nanmax(us[ok]):.0f} m/yr")
    np.savez_compressed(out, **res)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def plot(path, xlim, png):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import LogNorm
    z = np.load(path)
    names = list(z["names"])
    d = {n: z[f"{n}0"] for n in ("x", "y", "zs", "zb", "gm", "bed", "us", "ub")}
    x, y = d["x"] / 1e3, d["y"] / 1e3
    tri = mtri.Triangulation(x, y)
    sel = np.ones(len(x), bool) if xlim is None else (x >= xlim[0] - 5) & (x <= xlim[1] + 5)
    H = d["zs"] - d["zb"]
    fig, ax = plt.subplots(5, 1, figsize=(13, 15), sharex=True, constrained_layout=True)
    pos = lambda v: np.clip(v, 1.0, None)
    panels = [("surface speed (m/yr)", pos(d["us"]), "viridis", LogNorm(1, max(10, np.nanmax(d["us"][sel])))),
              ("ice thickness (m)", H, "cividis", None),
              ("ice base Zb (m)", d["zb"], "terrain", None),
              ("basal speed (m/yr)", pos(d["ub"]), "viridis", LogNorm(1, max(10, np.nanmax(d["ub"][sel])))),
              ("GroundedMask (1 grounded, -1 floating)", d["gm"], "coolwarm_r", None)]
    for a, (t, v, cm, norm) in zip(ax, panels):
        kw = dict(norm=norm) if norm else dict(vmin=np.nanmin(v[sel]), vmax=np.nanmax(v[sel]))
        im = a.tripcolor(tri, v, shading="gouraud", cmap=cm, **kw)
        fig.colorbar(im, ax=a, pad=0.01)
        a.set_title(t, loc="left", fontsize=10)
        a.set_ylabel("y (km)")
        a.set_aspect("auto")
    # the grounding line of every state
    for k, n in enumerate(names):
        xk, yk, gk = z[f"x{k}"] / 1e3, z[f"y{k}"] / 1e3, z[f"gm{k}"]
        tk = tri if k == 0 else mtri.Triangulation(xk, yk)
        for a in ax:
            cs = a.tricontour(tk, gk, levels=[0.0], colors="k" if k == 0 else "w", linewidths=1.2 if k == 0 else 1.0,
                              linestyles="solid" if k == 0 else "dashed")
        segs = [s for s in cs.allsegs[0] if len(s)]
        if segs:
            gx = np.concatenate(segs)[:, 0]
            print(f"{n}: grounding line x {gx.min():.2f} .. {gx.max():.2f} km (waviness {gx.max() - gx.min():.2f} km)")
    if xlim is not None:
        ax[-1].set_xlim(*xlim)
    ax[-1].set_xlabel("x (km)")
    fig.suptitle(f"{names[0]}" + (f"   (dashed white: {', '.join(names[1:])})" if len(names) > 1 else ""))
    if png:
        fig.savefig(png, dpi=130)
        print(f"wrote {png}")
    else:
        plt.show()


def plot3d(path, xlim, out, fs_km, periodic, exag, arrows):
    """an interactive 3D view (plotly.js from its CDN, opened in any browser): the ice surface and base coloured by
    speed, the bed, a vertical slice along y = width/2 coloured by speed, arrows of the surface flow, the grounding
    line (black) and the boundary between the full-Stokes region and the SSA shelf (red line on the surface and the
    base, and a red curtain through the ice)."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.tri as mtri
    import matplotlib.pyplot as plt
    z = np.load(path)
    d = {n: z[f"{n}0"] for n in ("x", "y", "zs", "zb", "gm", "bed", "us", "ub", "sux", "suy", "bux", "buy", "gld")}
    x, y = d["x"], d["y"]
    W = y.max() - y.min()
    tri = mtri.Triangulation(x / 1e3, y / 1e3)
    # the full-Stokes / SSA boundary: floating ice more than fs_km beyond the grounding line is the SSA shelf (as
    # GLDistance: distance to the grounding line, wrapped across the periodic sides). Elmer's own GLDistance when
    # the state has it, else rebuilt from the grounded mask (the grounding line = the GroundedMask 0 contour).
    fig0, a0 = plt.subplots()
    gls = [s_ for s_ in a0.tricontour(tri, d["gm"], levels=[0.0]).allsegs[0] if len(s_) > 1]
    if np.isfinite(d["gld"]).sum() > 0.9 * len(x):
        dist = d["gld"] / 1e3
        src = "Elmer's GLDistance"
    else:
        pts = np.concatenate(gls)
        if periodic:
            pts = np.concatenate([pts, pts + [0, W / 1e3], pts - [0, W / 1e3]])
        dist = np.zeros(len(x))
        fl = np.where(d["gm"] < -0.5)[0]
        try:
            from scipy.spatial import cKDTree
            dist[fl] = cKDTree(pts).query(np.c_[x[fl], y[fl]] / 1e3)[0]
        except ImportError:
            for c in np.array_split(fl, max(1, len(fl) // 2000)):
                q = np.c_[x[c], y[c]] / 1e3
                dist[c] = np.sqrt(((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(1)
        src = "rebuilt from the grounded mask"
    dfl = np.where(d["gm"] < -0.5, dist, 0.0)
    ifs = [s_ for s_ in a0.tricontour(tri, dfl, levels=[fs_km]).allsegs[0] if len(s_) > 1]
    plt.close(fig0)
    fzs = mtri.LinearTriInterpolator(tri, d["zs"])
    fzb = mtri.LinearTriInterpolator(tri, d["zb"])
    gx = np.concatenate(gls)[:, 0] if gls else np.array([np.nan])
    ix = np.concatenate(ifs)[:, 0] if ifs else np.array([np.nan])
    print(f"grounding line x {np.nanmin(gx):.2f} .. {np.nanmax(gx):.2f} km; full-Stokes / SSA boundary ({fs_km:g} km "
          f"beyond it, {src}) x {np.nanmin(ix):.2f} .. {np.nanmax(ix):.2f} km")

    # the window
    keep = np.ones(len(x), bool) if xlim is None else (x / 1e3 >= xlim[0]) & (x / 1e3 <= xlim[1])
    t = tri.triangles[keep[tri.triangles].all(1)]
    used = np.unique(t)
    ren = -np.ones(len(x), int); ren[used] = np.arange(len(used))
    I, J, K = (ren[t[:, c]].tolist() for c in range(3))
    X, Y = (x[used] / 1e3).round(3), (y[used] / 1e3).round(3)
    zsc = lambda v: (np.asarray(v) * exag / 1e3).round(4)   # km, exaggerated
    lg = lambda v: np.log10(np.clip(v, 1.0, None)).round(3)
    smax = float(np.log10(max(10.0, np.nanmax(d["us"][keep]))))
    cb = dict(title="speed (m/yr)", tickvals=list(range(0, int(smax) + 1)), ticktext=[f"{10 ** i:g}" for i in
              range(0, int(smax) + 1)])
    vel = lambda u, v: [f"{s_:.0f} m/yr" for s_ in np.hypot(u, v)]
    traces = [
        dict(type="mesh3d", name="ice surface", x=X.tolist(), y=Y.tolist(), z=zsc(d["zs"][used]).tolist(), i=I, j=J,
             k=K, intensity=lg(d["us"][used]).tolist(), colorscale="Viridis", cmin=0, cmax=smax, colorbar=cb,
             text=vel(d["sux"][used], d["suy"][used]), hoverinfo="text+name", flatshading=False, showlegend=True),
        dict(type="mesh3d", name="ice base", x=X.tolist(), y=Y.tolist(), z=zsc(d["zb"][used]).tolist(), i=I, j=J,
             k=K, intensity=lg(d["ub"][used]).tolist(), colorscale="Viridis", cmin=0, cmax=smax, showscale=False,
             text=vel(d["bux"][used], d["buy"][used]), hoverinfo="text+name", opacity=1.0, showlegend=True),
        dict(type="mesh3d", name="bed", x=X.tolist(), y=Y.tolist(), z=zsc(d["bed"][used] - 2.0).tolist(), i=I, j=J,
             k=K, color="rgb(150,130,110)", opacity=0.6, hoverinfo="skip", showlegend=True),
    ]

    def lines(segs, surf, name, color, width, show):
        xs, ys, zz = [], [], []
        for sg in segs:
            if xlim is not None:
                sg = sg[(sg[:, 0] >= xlim[0]) & (sg[:, 0] <= xlim[1])]
            if len(sg) < 2:
                continue
            zv = np.asarray(surf(sg[:, 0], sg[:, 1]).filled(np.nan))
            xs += sg[:, 0].round(3).tolist() + [None]; ys += sg[:, 1].round(3).tolist() + [None]
            zz += zsc(zv).tolist() + [None]
        return dict(type="scatter3d", mode="lines", name=name, x=xs, y=ys, z=[None if v is None or not np.isfinite(v)
                    else v for v in zz], line=dict(color=color, width=width), showlegend=show, legendgroup=name)
    up = lambda f: (lambda a, b: f(a, b) + 5.0)                   # just above the surface, so the line shows
    traces += [lines(ifs, up(fzs), f"full-Stokes | SSA boundary ({fs_km:g} km beyond the GL)", "red", 9, True),
               lines(ifs, fzb, f"full-Stokes | SSA boundary ({fs_km:g} km beyond the GL)", "red", 9, False),
               lines(gls, up(fzs), "grounding line", "black", 6, True),
               lines(gls, fzb, "grounding line", "black", 6, False)]
    # the red curtain: the boundary through the ice thickness
    cx, cy, cz, ci, cj, ck = [], [], [], [], [], []
    for sg in ifs:
        if xlim is not None:
            sg = sg[(sg[:, 0] >= xlim[0]) & (sg[:, 0] <= xlim[1])]
        if len(sg) < 2:
            continue
        ztop = np.asarray(fzs(sg[:, 0], sg[:, 1]).filled(np.nan))
        zbot = np.asarray(fzb(sg[:, 0], sg[:, 1]).filled(np.nan))
        f_ = np.isfinite(ztop) & np.isfinite(zbot)
        sg, ztop, zbot = sg[f_], ztop[f_], zbot[f_]
        if len(sg) < 2:
            continue
        n0 = len(cx)
        for p_, (a, b) in enumerate(zip(ztop, zbot)):
            cx += [sg[p_, 0]] * 2; cy += [sg[p_, 1]] * 2; cz += [a * exag / 1e3, b * exag / 1e3]
        for p_ in range(len(sg) - 1):
            q = n0 + 2 * p_
            ci += [q, q + 1]; cj += [q + 1, q + 3]; ck += [q + 2, q + 2]
    traces.append(dict(type="mesh3d", name="full-Stokes | SSA interface", x=np.round(cx, 3).tolist(),
                       y=np.round(cy, 3).tolist(), z=np.round(cz, 4).tolist(),
                       i=ci, j=cj, k=ck, color="red", opacity=0.35, hoverinfo="name", showlegend=True))
    # the vertical slice along y = width/2: speed through the ice
    sx, sz, su = z["slx0"], z["slz0"], z["slu0"]
    ysl = float(z["sly0"][0])
    ks = np.ones(len(sx), bool) if xlim is None else (sx / 1e3 >= xlim[0]) & (sx / 1e3 <= xlim[1])
    sx, sz, su = sx[ks], sz[ks], su[ks]
    if len(sx) > 3:
        zb_s = np.asarray(fzb(sx / 1e3, np.full(len(sx), ysl / 1e3)).filled(np.nan))
        zs_s = np.asarray(fzs(sx / 1e3, np.full(len(sx), ysl / 1e3)).filled(np.nan))
        zeta = np.clip((sz - zb_s) / np.maximum(zs_s - zb_s, 1.0), 0, 1)
        ok = np.isfinite(zeta)
        st = mtri.Triangulation(sx[ok] / 1e3, zeta[ok] * 20.0).triangles
        sp = np.hypot(su[ok, 0], su[ok, 1])
        traces.append(dict(type="mesh3d", name=f"slice y = {ysl / 1e3:.1f} km", x=(sx[ok] / 1e3).round(3).tolist(),
                           y=[round(ysl / 1e3, 3)] * int(ok.sum()), z=zsc(sz[ok]).tolist(), i=st[:, 0].tolist(),
                           j=st[:, 1].tolist(), k=st[:, 2].tolist(), intensity=lg(sp).tolist(), colorscale="Viridis",
                           cmin=0, cmax=smax, showscale=False, text=[f"{v:.0f} m/yr" for v in sp],
                           hoverinfo="text+name", showlegend=True))
    # arrows: the surface flow direction (length: log speed), on a random subset of the surface nodes
    if arrows > 0:
        rng = np.random.default_rng(0)
        sel = rng.choice(used, size=min(arrows, len(used)), replace=False)
        sp = np.hypot(d["sux"][sel], d["suy"][sel])
        L = 0.6 * np.log10(np.clip(sp, 1.0, None)) / max(smax, 1.0) * (np.ptp(X) / 40)
        ex, ey = d["sux"][sel] / np.maximum(sp, 1e-9), d["suy"][sel] / np.maximum(sp, 1e-9)
        ax_, ay_, az_ = [], [], []
        for p_ in range(len(sel)):
            x0, y0, z0 = x[sel[p_]] / 1e3, y[sel[p_]] / 1e3, d["zs"][sel[p_]] * exag / 1e3 + 0.01
            ax_ += [x0, x0 + L[p_] * ex[p_], None]; ay_ += [y0, y0 + L[p_] * ey[p_], None]; az_ += [z0, z0, None]
        traces.append(dict(type="scatter3d", mode="lines", name="surface flow (arrow length ~ log speed)",
                           x=ax_, y=ay_, z=az_, line=dict(color="white", width=3), hoverinfo="skip"))
    xr = (np.ptp(X), np.ptp(Y))
    layout = dict(title=f"{z['names'][0]}: speed, grounding line (black), full-Stokes | SSA boundary (red); "
                        f"vertical exaggeration x{exag:g}",
                  scene=dict(xaxis=dict(title="x (km)"), yaxis=dict(title="y (km)"),
                             zaxis=dict(title=f"z (km x {exag:g})"), aspectmode="manual",
                             aspectratio=dict(x=xr[0] / xr[1], y=1.0, z=0.4), camera=dict(eye=dict(x=-1.2, y=-1.6,
                             z=0.9))),
                  legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="white")
    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>3D state</title>"
            "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body style='margin:0'>"
            "<div id='p' style='width:100vw;height:100vh'></div><script>"
            f"Plotly.newPlot('p', {json.dumps(traces, allow_nan=False, default=float)}, {json.dumps(layout)});"
            "</script></body></html>")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB): open it in a browser")


def main():
    if len(sys.argv) == 1:                       # pressed Run
        return plot(FILE, XLIM, PNG)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("states", nargs="+")
    e.add_argument("--out", required=True)
    p = sp.add_parser("plot")
    p.add_argument("file")
    p.add_argument("--xlim", nargs=2, type=float, default=XLIM)
    p.add_argument("--all", action="store_true", help="the whole domain")
    p.add_argument("--png")
    q = sp.add_parser("plot3d")
    q.add_argument("file")
    q.add_argument("--xlim", nargs=2, type=float, default=XLIM)
    q.add_argument("--all", action="store_true", help="the whole domain")
    q.add_argument("--out", default="state3d.html")
    q.add_argument("--fs-ssa", type=float, default=10.0, help="km: the full-Stokes region beyond the grounding line")
    q.add_argument("--walls", action="store_true", help="no periodic wrap of the distance (walled sides)")
    q.add_argument("--exag", type=float, default=20.0, help="vertical exaggeration")
    q.add_argument("--arrows", type=int, default=600, help="surface flow arrows (0: none)")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.states, a.out)
    elif a.cmd == "plot3d":
        plot3d(a.file, None if a.all else a.xlim, a.out, a.fs_ssa, not a.walls, a.exag, a.arrows)
    else:
        plot(a.file, None if a.all else a.xlim, a.png)


if __name__ == "__main__":
    main()
