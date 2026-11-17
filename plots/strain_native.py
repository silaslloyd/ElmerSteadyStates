#!/usr/bin/env python3
"""The maps of plots/base_strain.py for ONE frame, at full resolution: on the model's own base nodes (no regridding),
drawn as one flat triangulated surface in true aspect (x and y on one scale, as plot_state_3d.py --true-aspect), with
one dropdown over every map of the base and of the surface ("base: e_xx", ..., "surface: e_xx", ...). The grounding
line in black. What the maps are, and the approximations they test: see plots/base_strain.py.

    python3 plots/strain_native.py FRAMES.npz [--t YEARS | --frame K] [--x XMIN XMAX] [--shear-factor F]
                                   [--keep-shelf] [--out FILE.html]

Default: the last frame, the whole x range, grounded ice only. The page holds every map's nodal values once
(float32), so its size is about 36 maps x the nodes in the x window x ~5 bytes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_strain import Joined, dominant_terms, frame_fields, opt  # noqa: E402


def main():
    args = sys.argv[1:]
    args, frame = opt(args, "--frame", conv=int)
    args, tsel = opt(args, "--t")
    args, xwin = opt(args, "--x", n=2)
    args, sf = opt(args, "--shear-factor", default=1.0)
    args, out = opt(args, "--out", conv=str)
    grounded_only = "--keep-shelf" not in args
    args = [a for a in args if a != "--keep-shelf"]
    if len(args) != 1:
        raise SystemExit(__doc__)
    path = args[0]
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator
    import plotly.graph_objects as go

    d = Joined([path])
    times = d["frame_times"]
    k = (int(np.argmin(np.abs(times - tsel))) if tsel is not None else
         (frame % len(times) if frame is not None else len(times) - 1))
    t = float(times[k])
    x, y = d["base_x"].astype(float), d["base_y"].astype(float)
    tx, ty = d["top_x"].astype(float), d["top_y"].astype(float)
    same = len(tx) == len(x) and np.allclose(tx, x) and np.allclose(ty, y)
    tt = None if same else Delaunay(np.c_[tx, ty])

    def interp_top(k_):
        f = (lambda a: a.astype(float)) if same else (lambda a: LinearNDInterpolator(tt, a.astype(float))(x, y))
        s = f(d["top_z"][k_])
        return f(d["top_vx"][k_]), f(d["top_vy"][k_]), f(d["top_vz"][k_]), s - d["base_z"][k_].astype(float), s

    tri = Delaunay(np.c_[x, y])
    maps = []                                      # (label, nodal values, kind)
    for side in ("base", "surface"):
        for lab, f, kind in frame_fields(d, k, x, y, tri, sf, interp_top, grounded_only, side):
            maps.append((f"{side}: {lab}", np.asarray(f, float), kind))
    # the drawn triangles: in the x window, all three nodes with values (grounded unless --keep-shelf)
    x0, x1 = (xwin if xwin else (x.min() / 1e3, x.max() / 1e3))
    T = tri.simplices
    ok = np.isfinite(maps[0][1])[T].all(1)
    cx = x[T].mean(1) / 1e3
    ok &= (cx >= x0) & (cx <= x1)
    # Delaunay of the nodes bridges the jagged periodic sides with slivers: drop triangles much larger than the mesh's
    L = np.max([np.hypot(x[T[:, a]] - x[T[:, b]], y[T[:, a]] - y[T[:, b]]) for a, b in ((0, 1), (1, 2), (2, 0))], 0)
    ok &= L < 6.0 * np.median(L[ok]) if ok.any() else ok
    T = T[ok]
    used = np.unique(T)
    re = -np.ones(len(x), np.int64)
    re[used] = np.arange(len(used))
    I, J, K = re[T[:, 0]], re[T[:, 1]], re[T[:, 2]]
    vx, vy = x[used] / 1e3, y[used] / 1e3
    print(f"t {t:g} yr: {len(used)} nodes, {len(T)} triangles in x {x0:g} .. {x1:g} km, {len(maps)} maps")

    # colour ranges as base_strain.py: one per kind, pooled over the maps of that kind
    def pooled(kind):
        z = np.concatenate([m[1][used] for m in maps if m[2] == kind])
        return z[np.isfinite(z)]
    zs = pooled("signed")
    a_s = float(np.percentile(np.abs(zs), 99)) if zs.size else 1.0
    zl = pooled("log")
    lo_l, hi_l = (float(v) for v in np.percentile(zl, [1, 99.5])) if zl.size else (-6.0, 0.0)
    names = [n_ for n_, _ in dominant_terms(*([np.zeros(1)] * 7))]
    cols = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e"]
    cat_cs = []
    for i, c in enumerate(cols[:len(names)]):
        cat_cs += [[i / len(names), c], [(i + 1) / len(names), c]]

    def style(kind):
        if kind == "signed":
            return dict(colorscale="RdBu_r", cmin=-a_s, cmax=a_s, cmid=0, title="1/yr")
        if kind == "log":
            return dict(colorscale="Viridis", cmin=lo_l, cmax=hi_l, title="log10 1/yr")
        if kind == "share":
            return dict(colorscale="Cividis", cmin=0, cmax=1, title="0 SIA .. 1 SSA")
        if kind == "ratio":
            return dict(colorscale="RdBu_r", cmin=-3, cmax=1, cmid=0, title="log10 ratio")
        if kind == "nh":
            return dict(colorscale="RdBu_r", cmin=-4, cmax=0, cmid=-2, title="log10 of rho g H")
        return dict(colorscale=cat_cs, cmin=-0.5, cmax=len(names) - 0.5, title="dominant")

    def values(f, kind):
        v = f[used].astype(np.float32)
        st = style(kind)
        # nodes without a value inside drawn triangles (e.g. no vertical shear for the ratio): below the range, grey
        return np.where(np.isfinite(v), v, np.float32(st["cmin"] - 1e6)).astype(np.float32)

    def restyle(lab, f, kind):
        st = style(kind)
        cs = st["colorscale"]
        if isinstance(cs, str):
            cs = cs                                        # named scale: grey is set by cmin trick via lowest colour
        cb = dict(title=st["title"])
        if kind == "nh":
            cb.update(tickvals=[-4, -3, -2, -1, 0], ticktext=["1e-4", "1e-3", "1%", "10%", "100%"])
        if kind == "cat":
            cb.update(tickvals=list(range(len(names))), ticktext=names)
        u = dict(intensity=[values(f, kind)], colorscale=[cs], cmin=[st["cmin"]], cmax=[st["cmax"]],
                 colorbar=[cb])
        if "cmid" in st:
            u["cmid"] = [st["cmid"]]
        return u

    lab0, f0, k0 = maps[0]
    r0 = restyle(lab0, f0, k0)
    mesh = go.Mesh3d(x=vx, y=vy, z=np.zeros(len(vx), np.float32), i=I, j=J, k=K, intensity=r0["intensity"][0],
                     intensitymode="vertex", colorscale=r0["colorscale"][0], cmin=r0["cmin"][0], cmax=r0["cmax"][0],
                     colorbar=r0["colorbar"][0], flatshading=True, lighting=dict(ambient=1, diffuse=0, specular=0),
                     hovertemplate="%{intensity:.3g}<br>x %{x:.1f} y %{y:.1f} km<extra></extra>", name="map")
    traces = [mesh]
    # the grounding line: the mask's 0 contour on the base nodes
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        gm = d["base_mask"][k].astype(float)
        cs_ = plt.tricontour(mtri.Triangulation(x / 1e3, y / 1e3, tri.simplices), gm, levels=[0.0])
        gxs, gys = [], []
        for seg in cs_.allsegs[0]:
            sel = (seg[:, 0] >= x0) & (seg[:, 0] <= x1)
            if sel.sum() > 1:
                gxs += list(seg[sel, 0]) + [None]
                gys += list(seg[sel, 1]) + [None]
        plt.close("all")
        if gxs:
            traces.append(go.Scatter3d(x=gxs, y=gys, z=[0.0 if v is not None else None for v in gxs], mode="lines",
                                       line=dict(color="black", width=4), hoverinfo="skip", name="grounding line"))
    except Exception as e:
        print(f"no grounding line drawn ({e})")
    buttons = []
    for lab, f, kind in maps:
        buttons.append(dict(label=lab, method="restyle", args=[restyle(lab, f, kind), [0]]))
    wx, wy = x1 - x0, (vy.max() - vy.min())
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{os.path.basename(path)}, t {t:,.0f} yr: strain rates on the model's nodes (dropdown: base / surface "
              f"maps; du/dz = {sf:g} x (u_s - u_b)/H)",
        updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.08, xanchor="left", yanchor="bottom",
                          active=0)],
        scene=dict(aspectmode="manual", aspectratio=dict(x=1.0, y=float(wy / wx), z=0.02),
                   xaxis=dict(title="x (km)", range=[x0, x1]), yaxis=dict(title="y (km)"),
                   zaxis=dict(visible=False),
                   camera=dict(projection=dict(type="orthographic"), eye=dict(x=0, y=0, z=1.5),
                               up=dict(x=0, y=1, z=0))),
        template="plotly_white", margin=dict(t=90, l=10, r=10, b=10), showlegend=False,
        annotations=[dict(x=1, y=1.08, xref="paper", yref="paper", xanchor="right", yanchor="bottom",
                          showarrow=False, font=dict(size=11),
                          text="SIA: du/dz, dv/dz only | SSA: e_xx, e_yy, e_xy | Blatter-Pattyn: all but dw/dx, dw/dy")])
    out = out or os.path.splitext(path)[0] + f"_strain_native_t{t:.0f}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
