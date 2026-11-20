#!/usr/bin/env python3
"""The spreading test (configs/spread_*.toml, branch test/lava-spreading): two top-down maps of each state, the surface
horizontal speed sqrt(u^2 + v^2) (m/yr, from the Stokes solution at the surface nodes) and below it the thickness
above the background layer h = H - H0 (the lava), one plotly page with a
slider over the states (model-time order), a "speed: log / linear" button and a "mesh on / off" button (the surface
nodes' triangulation); the current's margin (H - H0 = --dh m,
default 10) drawn as a white line. The map: the surface nodes' values interpolated linearly (on their Delaunay
triangles) onto a regular grid of about one sample per screen pixel (--width, default 1200 px); equal aspect; the
window: the current's extent (H - H0 > --dh) over all states, padded by a quarter, within the box (--window XMIN XMAX
YMIN YMAX in m to set it).

With --3d: one 3D surface (the surface Zs, coloured by u_h; no thickness map), --vex the vertical exaggeration
(default 1: true aspect).

    python3 plots/spread_plot.py STATE.npz [STATE2.npz ...] [--config configs/spread_ice.toml] [--dh 10] [--3d] [--vex 1]
                                 [--width 1200] [--window XMIN XMAX YMIN YMAX] [--out FILE.html]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.statefile import find_name, load_state  # noqa: E402


def arg(args, name, default, conv=str):
    if name in args:
        k = args.index(name)
        v = conv(args[k + 1])
        del args[k:k + 2]
        return v
    return default


def main():
    args = sys.argv[1:]
    cfgf = arg(args, "--config", os.path.join(ROOT, "configs", "spread_ice.toml"))
    dh = arg(args, "--dh", 10.0, float)
    width = arg(args, "--width", 1200, int)
    three_d = "--3d" in args                       # the maps as 3D surfaces (Zs), coloured by u_h and by h
    args = [a_ for a_ in args if a_ != "--3d"]
    vex = arg(args, "--vex", 1.0, float)           # (--3d) the vertical exaggeration (default 1: true aspect)
    win = None
    if "--window" in args:
        k = args.index("--window")
        win = [float(v) for v in args[k + 1:k + 5]]
        del args[k:k + 5]
    out = arg(args, "--out", None)
    if not args:
        raise SystemExit(__doc__)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    c = tomllib.load(open(cfgf, "rb"))["case"]
    L, W = float(c["spread_length_km"]) * 1e3, float(c["width"])
    xc, yc = float(c.get("source_x_km", L / 2e3)) * 1e3, float(c.get("source_y_km", W / 2e3)) * 1e3
    H0 = float(c.get("spread_thickness", 100.0))
    import plotly.graph_objects as go
    rows = []
    for p in sorted(args):
        s = load_state(p)
        allc = {**s.diag, **s.frozen}
        cx, cy = allc[find_name(allc, "coordinate 1")], allc[find_name(allc, "coordinate 2")]
        X, Y = dict(zip(cx[0].tolist(), cx[1])), dict(zip(cy[0].tolist(), cy[1]))
        blk = {b.name.lower(): b for b in s.layout.blocks}
        sl = s.layout.slices()
        gs = blk["zs"].gids
        zs = s.x[sl[blk["zs"].name]]
        B = dict(zip(*allc[find_name(allc, "bedrock")]))
        x = np.array([X[g] for g in gs])
        y = np.array([Y[g] for g in gs])
        H = zs - np.array([B[g] for g in gs])
        # the horizontal speed at the surface nodes (m/yr), from the Stokes solution (NaN if the state has none)
        try:
            V = [dict(zip(*allc[find_name(allc, f"velocity {k}")])) for k in (1, 2)]
            U = np.hypot(*[np.array([v_.get(g, np.nan) for g in gs], float) for v_ in V])
        except KeyError:
            U = np.full(len(gs), np.nan)
        t = float(s.meta.get("time", np.nan))
        rows.append(dict(t=t, x=x, y=y, zs=zs, H=H, U=U, r=np.hypot(x - xc, y - yc), name=os.path.basename(p)))
    rows.sort(key=lambda r_: r_["t"])               # by model time (the static and the spreading phases' files)
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    def fmt_t(t):
        if not np.isfinite(t) or t == 0:
            return "0"
        if t >= 1000:
            return f"{t:,.0f} yr"
        if t >= 1:
            return f"{t:.3g} yr"
        if t >= 1 / 365.25:
            return f"{t * 365.25:.3g} d"
        if t >= 1 / 8766:
            return f"{t * 8766:.3g} h"
        return f"{t * 525960:.3g} min"
    # the window: the current (H - H0 > dh) over all states, padded by a quarter (at least 2 elements' worth), in the box
    if win is None:
        xs_ = np.concatenate([r_["x"][r_["H"] - H0 > dh] for r_ in rows] + [np.array([])])
        ys_ = np.concatenate([r_["y"][r_["H"] - H0 > dh] for r_ in rows] + [np.array([])])
        if len(xs_):
            # a quarter of the current's size, at least a tenth of the box (a young flow: not a 1 m window)
            pad = max(0.25 * max(xs_.max() - xs_.min(), ys_.max() - ys_.min()), 0.1 * max(L, W))
            win = [max(0.0, xs_.min() - pad), min(L, xs_.max() + pad), max(0.0, ys_.min() - pad), min(W, ys_.max() + pad)]
        else:
            win = [0.0, L, 0.0, W]
    x0, x1, y0, y1 = win
    # the map fitted into about (width - 260) x 700 px at equal aspect; a sample per pixel
    # two maps (the speed above, the lava's thickness below), each fitted into ~(width - 260) x 450 px at equal aspect
    ppm = min((width - 260) / (x1 - x0), 450.0 / (y1 - y0))
    nx = max(100, int(round((x1 - x0) * ppm)))
    ny = max(50, int(round((y1 - y0) * ppm)))
    gap = 90
    height = 2 * ny + gap + 230
    width = max(nx + 260, 600)
    gx, gy = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    GX, GY = np.meshgrid(gx, gy)
    allU = np.concatenate([r_["U"][np.isfinite(r_["U"]) & (r_["U"] > 0)] for r_ in rows] + [np.array([1e-6])])
    u_hi = float(np.log10(np.percentile(allU, 99.5)))
    u_lo = u_hi - 4.0
    u_lin_hi = float(np.percentile(allU, 99.5))
    h_hi = max(float(np.nanmax(r_["H"] - H0)) for r_ in rows)
    h_hi = h_hi if h_hi > 0 else 1.0
    if not three_d:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=gap / (height - 230),
                            subplot_titles=("surface horizontal speed u_h", "lava thickness h = H - H0 (m)"))
        yfrac = lambda r: 1 - ((r - 1) * (ny + gap) + ny / 2) / (height - 230)
        cb = lambda r, title: dict(title=dict(text=title, side="right"), lenmode="pixels", len=ny, y=yfrac(r),
                                   yanchor="middle", thickness=14)
        # per state: 0 log u, 1 linear u, 2 the margin on u, 3 the mesh on u, 4 h, 5 the margin on h, 6 the mesh on h
        per = 7
        for i, r_ in enumerate(rows):
            print(f"{r_['name']}: t {fmt_t(r_['t'])}, h {max(float(np.nanmax(r_['H'] - H0)), 0.0):.3f} m max, surface speed "
                  f"max {np.nanmax(r_['U']) if np.isfinite(r_['U']).any() else np.nan:.3g} m/yr")
            tri = Delaunay(np.c_[r_["x"], r_["y"]])
            Ug = LinearNDInterpolator(tri, r_["U"])(GX, GY)
            Hg = LinearNDInterpolator(tri, r_["H"] - H0)(GX, GY)
            vis = i == len(rows) - 1
            with np.errstate(divide="ignore", invalid="ignore"):
                lu = np.log10(Ug)
            hov = "x %{x:.1f} m, y %{y:.1f} m<br>"
            sx_ = tri.simplices
            e_ = np.unique(np.sort(np.r_[sx_[:, [0, 1]], sx_[:, [1, 2]], sx_[:, [0, 2]]], axis=1), axis=0)
            inw = lambda k: (r_["x"][k] >= x0) & (r_["x"][k] <= x1) & (r_["y"][k] >= y0) & (r_["y"][k] <= y1)
            e_ = e_[inw(e_[:, 0]) | inw(e_[:, 1])]
            ex = np.c_[r_["x"][e_[:, 0]], r_["x"][e_[:, 1]], np.full(len(e_), np.nan)].ravel().astype(np.float32)
            ey = np.c_[r_["y"][e_[:, 0]], r_["y"][e_[:, 1]], np.full(len(e_), np.nan)].ravel().astype(np.float32)
            margin = lambda: go.Contour(x=gx, y=gy, z=Hg.astype(np.float32), contours=dict(start=dh, end=dh, size=1.0,
                                        coloring="none"), line=dict(color="white", width=1.5), showscale=False,
                                        visible=vis, hoverinfo="skip")
            mesh = lambda: go.Scattergl(x=ex, y=ey, mode="lines", line=dict(color="rgba(80,200,255,0.55)", width=0.6),
                                        visible=False, hoverinfo="skip", showlegend=False)
            fig.add_trace(go.Heatmap(x=gx, y=gy, z=lu.astype(np.float32), colorscale="Inferno", zmin=u_lo, zmax=u_hi,
                                     colorbar=cb(1, "log10 u_h (m/yr)"), visible=vis,
                                     hovertemplate=hov + "log10 u_h %{z:.2f}<extra></extra>"), row=1, col=1)
            fig.add_trace(go.Heatmap(x=gx, y=gy, z=Ug.astype(np.float32), colorscale="Inferno", zmin=0.0, zmax=u_lin_hi,
                                     colorbar=cb(1, "u_h (m/yr)"), visible=False,
                                     hovertemplate=hov + "u_h %{z:.3g} m/yr<extra></extra>"), row=1, col=1)
            fig.add_trace(margin(), row=1, col=1)
            fig.add_trace(mesh(), row=1, col=1)
            fig.add_trace(go.Heatmap(x=gx, y=gy, z=Hg.astype(np.float32), colorscale="Viridis", zmin=0.0, zmax=h_hi,
                                     colorbar=cb(2, "h (m)"), visible=vis,
                                     hovertemplate=hov + "h %{z:.3f} m<extra></extra>"), row=2, col=1)
            fig.add_trace(margin(), row=2, col=1)
            fig.add_trace(mesh(), row=2, col=1)
        n = len(rows)
        others = [j for j in range(per * n) if j % per in (2, 4, 5)]  # the slider's; the speeds and meshes: the script's
        steps = [dict(label=fmt_t(r_["t"]), method="update",
                      args=[{"visible": [j // per == i for j in others]},
                            {"title.text": f"t = {fmt_t(r_['t'])} (white: the margin, h = {dh:g} m)"}, others])
                 for i, r_ in enumerate(rows)]
        for r in (1, 2):
            fig.update_xaxes(range=[x0, x1], constrain="domain", row=r, col=1)
            fig.update_yaxes(title_text="y (m)", range=[y0, y1], scaleanchor=f"x{'' if r == 1 else r}", scaleratio=1,
                             constrain="domain", row=r, col=1)
        fig.update_xaxes(title_text="x (m)", row=2, col=1)
        fig.update_layout(template="plotly_white", width=width, height=height, margin=dict(l=70, r=150, t=80, b=130),
                          title=f"t = {fmt_t(rows[-1]['t'])} (white: the margin, h = {dh:g} m)",
                          sliders=[dict(active=n - 1, currentvalue=dict(prefix="t = "), steps=steps, pad=dict(t=50))],
                          updatemenus=[dict(type="buttons", direction="left", x=1.0, xanchor="right", y=1.0, yanchor="bottom",
                                            pad=dict(t=0, b=10), showactive=False,
                                            buttons=[dict(label="speed: log / linear", method="skip", args=[None]),
                                                     dict(label="mesh on / off", method="skip", args=[None])])])
    else:
        # 3D: the surface Zs on the surface nodes' triangles within the window, coloured by u_h, the vertical
        # exaggerated by vex; the margin (h = dh) a white line on the surface
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.tri as mtri
        zlo = min(float(np.nanmin(r_["zs"] - r_["H"])) for r_ in rows)
        zhi = max(float(np.nanmax(r_["zs"])) for r_ in rows)
        zr = max(zhi - zlo, 1e-6)
        height = 800
        fig = go.Figure()
        per = 4                                    # per state: log u, linear u, the margin, the mesh
        for i, r_ in enumerate(rows):
            print(f"{r_['name']}: t {fmt_t(r_['t'])}, h {max(float(np.nanmax(r_['H'] - H0)), 0.0):.3f} m max, surface "
                  f"speed max {np.nanmax(r_['U']) if np.isfinite(r_['U']).any() else np.nan:.3g} m/yr")
            x_, y_, z_ = r_["x"], r_["y"], r_["zs"]
            h_ = r_["H"] - H0
            tri = Delaunay(np.c_[x_, y_]).simplices
            inw = (x_ >= x0) & (x_ <= x1) & (y_ >= y0) & (y_ <= y1)
            tri = tri[inw[tri].any(axis=1)]          # every triangle touching the window (a clean edge, not a jagged one)
            vis = i == len(rows) - 1
            with np.errstate(divide="ignore", invalid="ignore"):
                lu = np.log10(r_["U"])
            lu = np.where(np.isfinite(lu), lu, u_lo)
            common = dict(x=x_.astype(np.float32), y=y_.astype(np.float32), z=z_.astype(np.float32),
                          i=tri[:, 0], j=tri[:, 1], k=tri[:, 2], flatshading=False,
                          lighting=dict(ambient=0.75, diffuse=0.4, specular=0.05, roughness=0.9))
            hov = "x %{x:.1f} m, y %{y:.1f} m, z %{z:.2f} m<br>"
            # the margin: the contour h = dh on the triangles, lifted onto the surface
            mx, my, mz = [], [], []
            try:
                T_ = mtri.Triangulation(x_, y_, tri)
                import matplotlib.pyplot as plt
                f_, ax_ = plt.subplots()
                cs_ = ax_.tricontour(T_, h_, levels=[dh])
                zi = mtri.LinearTriInterpolator(T_, z_)
                for seg in cs_.allsegs[0]:
                    zz = np.asarray(zi(seg[:, 0], seg[:, 1]), float) + 0.003 * zr
                    mx += list(seg[:, 0]) + [np.nan]
                    my += list(seg[:, 1]) + [np.nan]
                    mz += list(zz) + [np.nan]
                plt.close(f_)
            except Exception as e:                       # noqa: BLE001
                print(f"  no margin line ({e})")
            e_ = np.unique(np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]], axis=1), axis=0)
            nan = np.full(len(e_), np.nan)
            ex = np.c_[x_[e_[:, 0]], x_[e_[:, 1]], nan].ravel().astype(np.float32)
            ey = np.c_[y_[e_[:, 0]], y_[e_[:, 1]], nan].ravel().astype(np.float32)
            ez = (np.c_[z_[e_[:, 0]], z_[e_[:, 1]], nan].ravel() + 0.002 * zr).astype(np.float32)
            margin = lambda sc: go.Scatter3d(x=mx, y=my, z=mz, mode="lines", line=dict(color="white", width=4),
                                             visible=vis, hoverinfo="skip", showlegend=False, scene=sc)
            mesh = lambda sc: go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="rgba(80,200,255,0.6)",
                                           width=1), visible=False, hoverinfo="skip", showlegend=False, scene=sc)
            fig.add_trace(go.Mesh3d(**common, intensity=lu.astype(np.float32), colorscale="Inferno", cmin=u_lo,
                                    cmax=u_hi, colorbar=dict(title="log10 u_h (m/yr)", len=0.8), visible=vis,
                                    hovertemplate=hov + "log10 u_h %{intensity:.2f}<extra></extra>", scene="scene"))
            fig.add_trace(go.Mesh3d(**common, intensity=r_["U"].astype(np.float32), colorscale="Inferno", cmin=0.0,
                                    cmax=u_lin_hi, colorbar=dict(title="u_h (m/yr)", len=0.8), visible=False,
                                    hovertemplate=hov + "u_h %{intensity:.3g} m/yr<extra></extra>", scene="scene"))
            fig.add_trace(margin("scene"))
            fig.add_trace(mesh("scene"))
        n = len(rows)
        others = [j for j in range(per * n) if j % per == 2]
        steps = [dict(label=fmt_t(r_["t"]), method="update",
                      args=[{"visible": [j // per == i for j in others]},
                            {"title.text": f"t = {fmt_t(r_['t'])}: the surface coloured by u_h; "
                                           f"vertical exaggeration {vex:g}"}, others])
                 for i, r_ in enumerate(rows)]
        sc = dict(xaxis=dict(title="x (m)", range=[x0, x1]), yaxis=dict(title="y (m)", range=[y0, y1]),
                  zaxis=dict(title="z (m)", range=[zlo, zhi + 0.02 * zr]), aspectmode="manual",
                  aspectratio=dict(x=1, y=(y1 - y0) / (x1 - x0), z=vex * zr / (x1 - x0)),
                  camera=dict(eye=dict(x=0.45, y=-0.95, z=0.6)))
        fig.update_layout(template="plotly_white", autosize=True, margin=dict(l=10, r=10, t=90, b=60),
                          scene=dict(**sc, domain=dict(x=[0, 0.92], y=[0.0, 1.0])),
                          title=f"t = {fmt_t(rows[-1]['t'])}: the surface coloured by u_h; vertical exaggeration {vex:g}",
                          sliders=[dict(active=n - 1, currentvalue=dict(prefix="t = "), steps=steps, pad=dict(t=30))],
                          updatemenus=[dict(type="buttons", direction="left", x=1.0, xanchor="right", y=1.0,
                                            yanchor="bottom", pad=dict(t=0, b=10), showactive=False,
                                            buttons=[dict(label="speed: log / linear", method="skip", args=[None]),
                                                     dict(label="mesh on / off", method="skip", args=[None])])])
    meshoff = "[3]" if three_d else "[3, 6]"       # the mesh traces' offsets in a state's block
    js = ("var gd = document.getElementById('{plot_id}'); var per = %d, n = %d, cur = n - 1, mode = 'log', mesh = false;\n"
          "var meshoff = %s;\n"
          "function spd() { var idx = [], vis = [];\n"
          "  for (var i = 0; i < n; i++) { idx.push(i * per, i * per + 1);\n"
          "    vis.push(i == cur && mode == 'log', i == cur && mode == 'lin');\n"
          "    for (var k = 0; k < meshoff.length; k++) { idx.push(i * per + meshoff[k]); vis.push(i == cur && mesh); } }\n"
          "  Plotly.restyle(gd, {visible: vis}, idx); }\n"
          "gd.on('plotly_sliderchange', function (e) { cur = e.step._index; spd(); });\n"
          "gd.on('plotly_buttonclicked', function (e) {\n"
          "  if (e.button.label.indexOf('mesh') == 0) { mesh = !mesh; } else { mode = (mode == 'log') ? 'lin' : 'log'; }\n"
          "  spd(); });\n") % (per, n, meshoff)
    out = out or os.path.join(os.path.dirname(os.path.abspath(sorted(args)[-1])), "spread.html")
    if three_d:     # the whole browser window (resizes with it)
        fig.write_html(out, include_plotlyjs="cdn", post_script=js, default_width="100%", default_height="96vh",
                       config=dict(responsive=True))
    else:
        fig.write_html(out, include_plotlyjs="cdn", post_script=js)
    print(f"-> {out} (map {nx} x {ny} samples, window x {x0:.0f} .. {x1:.0f} m, y {y0:.0f} .. {y1:.0f} m)")


if __name__ == "__main__":
    main()
