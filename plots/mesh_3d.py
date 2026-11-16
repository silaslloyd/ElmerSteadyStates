#!/usr/bin/env python3
"""A footprint mesh on its own, as an interactive plotly page (like plots/plot_state_3d.py, but only the mesh):
the triangles coloured by element size (mean edge length), their edges, and, with a state, the grounding line (red)
and the cold/temperate transition on the grounded bed (magenta), the mesh drawn at the ice base of that state.

Two steps, since the meshes are on tethys and plotly on pilsen:

    python3 plots/mesh_3d.py pack MESHDIR OUT.npz [--state STATE.npz]     (tethys: numpy only; a few MB)
    python3 plots/mesh_3d.py plot IN.npz [IN2.npz ..] [OUT.html] [--flat] [--edges] [--zoom X0 X1 Y0 Y1 (km)]
        (pilsen: plotly; several meshes: tiled as periodic copies of the sector (mesh k rotated by k times the sector angle), in the order given; OUT default
         <first>_mesh3d.html, or <first>_cmp_mesh3d.html for several)

The triangles are filled by element size (flat shading, so each one shows); --edges also draws every edge as a line
(large: best with --zoom).

MESHDIR: the mesh directory in the case dir, e.g. work/<dir>/case/31811m009 (or test250); STATE: a state on that
mesh, e.g. <work>/<mesh>_start.npz (the state it was made from, interpolated onto it). --flat: z = 0 instead of the
ice base (a plain map you can still rotate and zoom).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def pack(meshdir, out, state=None):
    from ess.remesh import base_values, read_footprint
    _, x, y, tri = read_footprint(meshdir)
    nb = len(x)
    rec = dict(x=x.astype(np.float32), y=y.astype(np.float32), tri=tri.astype(np.int32),
               name=os.path.basename(os.path.normpath(meshdir)))
    if state:
        from ess.statefile import find_name, load_state
        s = load_state(state)
        f = {**s.diag, **s.frozen}
        blocks = s.layout.split(s.x)
        zb = next(((b.gids, blocks[b.name]) for b in s.layout.blocks if b.name.lower() == "zb"), None)
        if zb is not None:
            rec["zb"] = base_values(zb, nb).astype(np.float32)
        for nm, key in (("groundedmask", "gm"), ("coldtempmask", "ct")):
            try:
                rec[key] = base_values(f[find_name(f, nm)], nb).astype(np.float32)
            except KeyError:
                pass
        if "gm" in rec and np.isnan(rec["gm"]).all():
            raise SystemExit(f"{state}: not on this mesh ({nb} footprint nodes)")
        rec["state"] = state
        rec["time"] = float(s.meta.get("time", np.nan) or np.nan)
    np.savez_compressed(out, **rec)
    print(f"{meshdir}: {nb} nodes, {len(tri)} triangles -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def contour_lines(x, y, tri, v, level, mask=None):
    """the level line of v on the triangles (mask: triangles to leave out) as polyline x, y with None gaps"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    T = mtri.Triangulation(x, y, tri, mask=mask)
    fig, ax = plt.subplots()
    cs = ax.tricontour(T, v, levels=[level])
    X, Y = [], []
    for seg in cs.allsegs[0]:
        X += list(seg[:, 0]) + [None]
        Y += list(seg[:, 1]) + [None]
    plt.close(fig)
    return X, Y


def sector_angle(x, y):
    """a sector footprint's angle (radians): the span of the nodes on its outer arc (as plots/plot_state_3d.py)"""
    r, ph = np.hypot(x, y), np.arctan2(y, x)
    m = r > r.max() - 1.0
    return float(ph[m].max() - ph[m].min()) if m.sum() > 1 else float(np.ptp(ph))


def panel_traces(inp, scene, flat=False, zoom=None, edges=False, first=True, rot=0.0):
    """the traces of one mesh in the 3D scene `scene`, rotated by rot (radians) about the sector's apex: copy k of a
    periodic sector at k times its angle, so the copies meet at the periodic sides; and its title"""
    import plotly.graph_objects as go
    d = np.load(inp, allow_pickle=True)
    x, y, tri = d["x"].astype(float) / 1e3, d["y"].astype(float) / 1e3, d["tri"].astype(np.int64)
    name = str(d["name"])
    nb = len(x)
    ntri = len(tri)
    z = np.zeros(nb) if flat or "zb" not in d.files else np.nan_to_num(d["zb"].astype(float))
    if zoom:
        keep = ((x[tri] > zoom[0]) & (x[tri] < zoom[1]) & (y[tri] > zoom[2]) & (y[tri] < zoom[3])).any(1)
        tri = tri[keep]
    if isinstance(rot, tuple):                    # a strip: translated across the flow (a sector of infinite radius)
        y = y + rot[1] / 1e3
    elif rot:
        x, y = np.cos(rot) * x - np.sin(rot) * y, np.sin(rot) * x + np.cos(rot) * y
    # element size: the mean edge length (m)
    L = [np.hypot(x[tri[:, a_]] - x[tri[:, b_]], y[tri[:, a_]] - y[tri[:, b_]]) for a_, b_ in ((0, 1), (1, 2), (2, 0))]
    hs = np.mean(L, 0) * 1e3
    lh = np.log10(np.clip(hs, 100, 1e4))
    ticks = [125, 250, 500, 1000, 2000, 5000]
    traces = [go.Mesh3d(x=x, y=y, z=z, i=tri[:, 0], j=tri[:, 1], k=tri[:, 2], intensity=lh, intensitymode="cell",
                        colorscale="Viridis", reversescale=True, cmin=np.log10(150), cmax=np.log10(6000),
                        showscale=first,
                        colorbar=dict(title="element size (m)", tickvals=list(np.log10(ticks)),
                                      ticktext=[str(t) for t in ticks]),
                        customdata=np.round(hs).astype(int), hovertemplate="size %{customdata} m<extra></extra>",
                        flatshading=True, lighting=dict(ambient=1, diffuse=0, specular=0), name="elements",
                        scene=scene)]
    nn = lambda a_: [None if np.isnan(v) else round(float(v), 3) for v in a_.ravel()]
    if edges:                                     # every edge as a line: large (~3 points per edge)
        e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
        e = np.unique(np.sort(e, 1), axis=0)
        zl = z + (0 if flat else 1.0)
        P = np.full((len(e), 3), np.nan)
        ex, ey, ez = P.copy(), P.copy(), P.copy()
        ex[:, :2], ey[:, :2], ez[:, :2] = x[e], y[e], zl[e]
        traces.append(go.Scatter3d(x=nn(ex), y=nn(ey), z=nn(ez), mode="lines", line=dict(color="#333333", width=1),
                                   hoverinfo="skip", name="edges", legendgroup="edges", showlegend=first,
                                   scene=scene))
    curves = []
    if "gm" in d.files:
        gm = np.nan_to_num(d["gm"].astype(float), nan=-1.0)
        X, Y = contour_lines(x, y, tri, gm, -0.5)
        curves.append(("grounding line", X, Y, "red"))
        if "ct" in d.files:
            ct = np.nan_to_num(d["ct"].astype(float), nan=-1.0)
            X, Y = contour_lines(x, y, tri, ct, 0.0, mask=~(gm[tri] > -0.5).all(1))
            curves.append(("cold/temperate transition (grounded)", X, Y, "magenta"))
    from scipy.interpolate import LinearNDInterpolator
    zi = None if flat or "zb" not in d.files else LinearNDInterpolator(np.c_[x, y], z)
    for lab, X, Y, col in curves:
        Xa = np.array([np.nan if v is None else v for v in X], float)
        Ya = np.array([np.nan if v is None else v for v in Y], float)
        Za = np.zeros_like(Xa) if zi is None else zi(Xa, Ya) + 3.0   # flat: on the mesh, z = 0
        traces.append(go.Scatter3d(x=nn(Xa), y=nn(Ya), z=nn(Za), mode="lines", line=dict(color=col, width=5),
                                   hoverinfo="skip", name=lab, legendgroup=lab, showlegend=first, scene=scene))
    title = f"{name}: {nb} nodes, {ntri} triangles" + (f", t {float(d['time']):g} yr" if "time" in d.files else "")
    keep_nodes = np.unique(tri)
    traces.append(go.Scatter3d(x=[float(x[keep_nodes].max()) + 15], y=[float(y[keep_nodes].mean())],
                               z=[0.0 if zi is None else float(np.nanmax(z)) + 50], mode="text", text=[title],
                               textposition="middle right", showlegend=False, hoverinfo="skip", scene=scene))
    asp = dict(x=3, y=3 * (np.ptp(y) / max(np.ptp(x), 1e-9)), z=0.02 if zi is None else 0.3)
    return traces, title, asp


def plot(inps, out=None, flat=False, zoom=None, edges=False):
    """one page, one view: the meshes as periodic copies of the sector, each rotated by the sector angle from the last"""
    import plotly.graph_objects as go
    d0 = np.load(inps[0], allow_pickle=True)
    x0, y0 = d0["x"].astype(float), d0["y"].astype(float)
    strip = float(y0.min()) > -1.0                # a strip (0 <= y <= W); a sector is symmetric about y = 0
    if strip:
        Wd = float(np.ptp(y0))
        print(f"strip width {Wd / 1e3:g} km: mesh k translated by k times it (they meet at the periodic sides)")
    else:
        A = sector_angle(x0, y0)                  # the copies tile periodically: copy k at k A
        print(f"sector angle {np.degrees(A):.3f} deg: mesh k rotated by k times it (they meet at the periodic sides)")
    traces, titles = [], []
    for k, f in enumerate(inps):
        tr, t_, _ = panel_traces(f, "scene", flat, zoom, edges, k == 0, rot=(0.0, k * Wd) if strip else k * A)
        traces += tr
        titles.append(t_)
    fig = go.Figure(traces)
    xs = np.concatenate([np.asarray([v for v in t_.x if v is not None], float) for t_ in traces
                         if getattr(t_, "x", None) is not None and t_.type == "mesh3d"])
    ys = np.concatenate([np.asarray(t_.y, float) for t_ in traces if t_.type == "mesh3d"])
    asp = dict(x=3, y=3 * float(np.ptp(ys)) / max(float(np.ptp(xs)), 1e-9), z=0.02 if flat else 0.3)
    fig.update_layout(title=" | ".join(titles) if len(titles) > 1 else titles[0], legend=dict(x=0, y=1),
                      scene=dict(aspectmode="manual", aspectratio=asp, xaxis_title="x (km)",
                                 yaxis_title="y (km)", zaxis_title="z (m)"))
    if flat:                                      # a fixed thin z range: the curves on the mesh, no stretched z
        fig.layout.scene.zaxis.update(range=[-1, 1], visible=False)
    out = out or os.path.splitext(inps[0])[0] + ("_cmp" if len(inps) > 1 else "") + "_mesh3d.html"
    fig.write_html(out, include_plotlyjs="cdn", config=dict(responsive=True), default_width="100%",
                   default_height="95vh")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


def main():
    a = sys.argv[1:]
    if len(a) < 2 or a[0] not in ("pack", "plot"):
        raise SystemExit(__doc__)
    if a[0] == "pack":
        st = None
        if "--state" in a:
            k = a.index("--state")
            st = a[k + 1]
            del a[k:k + 2]
        if len(a) < 3:
            raise SystemExit(__doc__)
        pack(a[1], a[2], st)
    else:
        flat, edges = "--flat" in a, "--edges" in a
        a = [v for v in a if v not in ("--flat", "--edges")]
        zoom = None
        if "--zoom" in a:
            k = a.index("--zoom")
            zoom = [float(v) for v in a[k + 1:k + 5]]
            del a[k:k + 5]
        inps = [v for v in a[1:] if v.endswith(".npz")]
        outs = [v for v in a[1:] if v.endswith(".html")]
        plot(inps, outs[0] if outs else None, flat, zoom, edges)


if __name__ == "__main__":
    main()
