#!/usr/bin/env python3
"""The spreading test (configs/spread_*.toml, branch test/lava-spreading): a top-down map of the surface horizontal
speed sqrt(u^2 + v^2) (m/yr, from the Stokes solution at the surface nodes) of each state, one plotly page with a
slider over the states (model-time order), a "speed: log / linear" button and a "mesh on / off" button (the surface
nodes' triangulation); the current's margin (H - H0 = --dh m,
default 10) drawn as a white line. The map: the surface nodes' values interpolated linearly (on their Delaunay
triangles) onto a regular grid of about one sample per screen pixel (--width, default 1200 px); equal aspect; the
window: the current's extent (H - H0 > --dh) over all states, padded by a quarter, within the box (--window XMIN XMAX
YMIN YMAX in m to set it).

    python3 plots/spread_plot.py STATE.npz [STATE2.npz ...] [--config configs/spread_ice.toml] [--dh 10]
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
    ppm = min((width - 260) / (x1 - x0), 700.0 / (y1 - y0))
    nx = max(100, int(round((x1 - x0) * ppm)))
    ny = max(50, int(round((y1 - y0) * ppm)))
    height = ny + 220
    width = max(nx + 260, 600)
    gx, gy = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    GX, GY = np.meshgrid(gx, gy)
    allU = np.concatenate([r_["U"][np.isfinite(r_["U"]) & (r_["U"] > 0)] for r_ in rows] + [np.array([1e-6])])
    u_hi = float(np.log10(np.percentile(allU, 99.5)))
    u_lo = u_hi - 4.0
    u_lin_hi = float(np.percentile(allU, 99.5))
    fig = go.Figure()
    per = 4                                    # per state: log speed, linear speed, the margin line, the mesh
    for i, r_ in enumerate(rows):
        print(f"{r_['name']}: t {fmt_t(r_['t'])}, H {r_['H'].min():.2f} .. {r_['H'].max():.2f} m, surface speed max "
              f"{np.nanmax(r_['U']) if np.isfinite(r_['U']).any() else np.nan:.3g} m/yr")
        tri = Delaunay(np.c_[r_["x"], r_["y"]])
        Ug = LinearNDInterpolator(tri, r_["U"])(GX, GY)
        Hg = LinearNDInterpolator(tri, r_["H"] - H0)(GX, GY)
        vis = i == len(rows) - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            lu = np.log10(Ug)
        hov = "x %{x:.1f} m, y %{y:.1f} m<br>"
        fig.add_trace(go.Heatmap(x=gx, y=gy, z=lu.astype(np.float32), colorscale="Inferno", zmin=u_lo, zmax=u_hi,
                                 colorbar=dict(title="log10 u_h (m/yr)"), visible=vis,
                                 hovertemplate=hov + "log10 u_h %{z:.2f}<extra></extra>"))
        fig.add_trace(go.Heatmap(x=gx, y=gy, z=Ug.astype(np.float32), colorscale="Inferno", zmin=0.0, zmax=u_lin_hi,
                                 colorbar=dict(title="u_h (m/yr)"), visible=False,
                                 hovertemplate=hov + "u_h %{z:.3g} m/yr<extra></extra>"))
        fig.add_trace(go.Contour(x=gx, y=gy, z=Hg.astype(np.float32), contours=dict(start=dh, end=dh, size=1.0,
                                 coloring="none"), line=dict(color="white", width=1.5), showscale=False, visible=vis,
                                 hoverinfo="skip"))
        # the mesh: the edges of the surface nodes' Delaunay triangulation (the footprint mesh's own triangles where
        # gmsh's are Delaunay, as they normally are) within the window
        sx_ = tri.simplices
        e_ = np.unique(np.sort(np.r_[sx_[:, [0, 1]], sx_[:, [1, 2]], sx_[:, [0, 2]]], axis=1), axis=0)
        inw = lambda k: (r_["x"][k] >= x0) & (r_["x"][k] <= x1) & (r_["y"][k] >= y0) & (r_["y"][k] <= y1)
        e_ = e_[inw(e_[:, 0]) | inw(e_[:, 1])]
        ex = np.c_[r_["x"][e_[:, 0]], r_["x"][e_[:, 1]], np.full(len(e_), np.nan)].ravel()
        ey = np.c_[r_["y"][e_[:, 0]], r_["y"][e_[:, 1]], np.full(len(e_), np.nan)].ravel()
        fig.add_trace(go.Scattergl(x=ex.astype(np.float32), y=ey.astype(np.float32), mode="lines",
                                   line=dict(color="rgba(80,200,255,0.55)", width=0.6), visible=False,
                                   hoverinfo="skip", showlegend=False))
    n = len(rows)
    others = [j for j in range(per * n) if j % per == 2]          # the margin lines: the slider's; the rest: the script's
    steps = [dict(label=fmt_t(r_["t"]), method="update",
                  args=[{"visible": [j // per == i for j in others]},
                        {"title.text": f"surface horizontal speed, t = {fmt_t(r_['t'])} (white: the margin, "
                                       f"H - H0 = {dh:g} m)"}, others]) for i, r_ in enumerate(rows)]
    fig.update_xaxes(title_text="x (m)", range=[x0, x1], constrain="domain")
    fig.update_yaxes(title_text="y (m)", range=[y0, y1], scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(template="plotly_white", width=width, height=height, margin=dict(l=70, r=150, t=80, b=130),
                      title=f"surface horizontal speed, t = {fmt_t(rows[-1]['t'])} (white: the margin, H - H0 = {dh:g} m)",
                      sliders=[dict(active=n - 1, currentvalue=dict(prefix="t = "), steps=steps, pad=dict(t=50))],
                      updatemenus=[dict(type="buttons", direction="left", x=1.0, xanchor="right", y=1.0, yanchor="bottom",
                                        pad=dict(t=0, b=10), showactive=False,
                                        buttons=[dict(label="speed: log / linear", method="skip", args=[None]),
                                                 dict(label="mesh on / off", method="skip", args=[None])])])
    js = ("var gd = document.getElementById('{plot_id}'); var per = %d, n = %d, cur = n - 1, mode = 'log', mesh = false;\n"
          "function spd() { var idx = [], vis = [];\n"
          "  for (var i = 0; i < n; i++) { idx.push(i * per, i * per + 1, i * per + 3);\n"
          "    vis.push(i == cur && mode == 'log', i == cur && mode == 'lin', i == cur && mesh); }\n"
          "  Plotly.restyle(gd, {visible: vis}, idx); }\n"
          "gd.on('plotly_sliderchange', function (e) { cur = e.step._index; spd(); });\n"
          "gd.on('plotly_buttonclicked', function (e) {\n"
          "  if (e.button.label.indexOf('mesh') == 0) { mesh = !mesh; } else { mode = (mode == 'log') ? 'lin' : 'log'; }\n"
          "  spd(); });\n") % (per, n)
    out = out or os.path.join(os.path.dirname(os.path.abspath(sorted(args)[-1])), "spread.html")
    fig.write_html(out, include_plotlyjs="cdn", post_script=js)
    print(f"-> {out} (map {nx} x {ny} samples, window x {x0:.0f} .. {x1:.0f} m, y {y0:.0f} .. {y1:.0f} m)")


if __name__ == "__main__":
    main()
