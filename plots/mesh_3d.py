#!/usr/bin/env python3
"""A footprint mesh on its own, as an interactive plotly page (like plots/plot_state_3d.py, but only the mesh):
the triangles coloured by element size (mean edge length), their edges, and, with a state, the grounding line (red)
and the cold/temperate transition on the grounded bed (magenta), the mesh drawn at the ice base of that state.

Two steps, since the meshes are on tethys and plotly on pilsen:

    python3 plots/mesh_3d.py pack MESHDIR OUT.npz [--state STATE.npz]     (tethys: numpy only; a few MB)
    python3 plots/mesh_3d.py plot IN.npz [IN2.npz ..] [OUT.html] [--flat] [--edges] [--zoom X0 X1 Y0 Y1 (km)]
        (pilsen: plotly; several meshes: one panel each, one above the other, the cameras linked; OUT default
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


def panel_traces(inp, scene, flat=False, zoom=None, edges=False, first=True):
    """the traces of one mesh in the 3D scene `scene`, and its title"""
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
        Za = np.zeros_like(Xa) + 2.0 if zi is None else zi(Xa, Ya) + 3.0
        traces.append(go.Scatter3d(x=nn(Xa), y=nn(Ya), z=nn(Za), mode="lines", line=dict(color=col, width=5),
                                   hoverinfo="skip", name=lab, legendgroup=lab, showlegend=first, scene=scene))
    title = f"{name}: {nb} nodes, {ntri} triangles" + (f", t {float(d['time']):g} yr" if "time" in d.files else "")
    asp = dict(x=3, y=3 * (np.ptp(y) / max(np.ptp(x), 1e-9)), z=0.3)
    return traces, title, asp


def plot(inps, out=None, flat=False, zoom=None, edges=False):
    """one page: each mesh in its own 3D panel, one above the other, the cameras linked"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    n = len(inps)
    parts = [panel_traces(f, "scene" if k == 0 else f"scene{k + 1}", flat, zoom, edges, k == 0)
             for k, f in enumerate(inps)]
    fig = make_subplots(rows=n, cols=1, specs=[[{"type": "scene"}] for _ in range(n)],
                        subplot_titles=[t for _, t, _ in parts], vertical_spacing=0.03)
    for k, (tr, _, asp) in enumerate(parts):
        for t_ in tr:
            fig.add_trace(t_, row=k + 1, col=1)
        fig.layout["scene" if k == 0 else f"scene{k + 1}"].update(
            aspectmode="manual", aspectratio=asp, xaxis_title="x (km)", yaxis_title="y (km)", zaxis_title="z (m)")
    fig.update_layout(legend=dict(x=0, y=1))
    # the panels' cameras linked: rotating or zooming one moves the others
    sync = None
    if n > 1:
        sync = """
(function () {
  var g = document.getElementsByClassName('plotly-graph-div')[0], busy = false;
  var S = %s;
  g.on('plotly_relayout', function (e) {
    if (busy) return; var cam = null, src = null;
    S.forEach(function (s) { if (e[s + '.camera']) { cam = e[s + '.camera']; src = s; } });
    if (!cam) return; var u = {}; S.forEach(function (s) { if (s !== src) u[s + '.camera'] = cam; });
    busy = true; Plotly.relayout(g, u).then(function () { busy = false; });
  });
})();""" % ([("scene" if k == 0 else f"scene{k + 1}") for k in range(n)],)
    out = out or os.path.splitext(inps[0])[0] + ("_cmp" if n > 1 else "") + "_mesh3d.html"
    fig.write_html(out, include_plotlyjs="cdn", config=dict(responsive=True), default_width="100%",
                   default_height=f"{95 * n}vh" if n > 1 else "95vh", post_script=sync)
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
