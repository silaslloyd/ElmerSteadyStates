#!/usr/bin/env python3
"""The spreading test (configs/spread_ice.toml): a vertical cross-section through the source centre (y = the source's
y, along x; --axis y for the section along y), temperature and viscosity, one plotly page with a slider over the
states (in model-time order) and buttons for the field:
  * temperature T (K);
  * log10 of the viscosity (Pa s), eta = viscosity * exp(Q/R (1/T - 1/Tref)) as the model has it (SpreadViscosity in
    elmer/ess_utils.lua: T capped to 150 .. T0), Q and Tref from the config, `viscosity` from parameters.lua.
Each extruded level of the mesh is interpolated onto the section line (linear on the footprint's triangles), then every
column onto a regular z grid (--nz levels): the colours are the model's values, nothing smoothed across levels.

    python3 plots/spread_section.py STATE.npz [STATE2.npz ...] [--config configs/spread_ice.toml] [--axis x|y]
                                    [--nx 300] [--nz 120] [--out FILE.html]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.case import eval_lua_parameters  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402


def arg(args, name, default, conv=str):
    if name in args:
        k = args.index(name)
        v = conv(args[k + 1])
        del args[k:k + 2]
        return v
    return default


def section(s, axis, c0, nx, L):
    """(s along the line (m), z grid, T grid) of the state s on the line {y = c0} (axis x) or {x = c0} (axis y)"""
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator
    allc = {**s.diag, **s.frozen}
    cx, cy, cz = (allc[find_name(allc, f"coordinate {k}")] for k in (1, 2, 3))
    blk = {b.name.lower(): b for b in s.layout.blocks}
    bT = blk["temperature"]
    g = bT.gids
    look = lambda c: np.interp(g, c[0], c[1]) if np.all(np.diff(c[0]) > 0) else np.array(
        [dict(zip(c[0].tolist(), c[1]))[k] for k in g.tolist()])
    X, Y, Z = look(cx), look(cy), look(cz)
    T = s.x[s.layout.slices()[bT.name]]
    col = np.unique(np.c_[np.round(X, 1), np.round(Y, 1)], axis=0, return_inverse=True)[1].ravel()
    ncol = col.max() + 1
    nl = len(g) // ncol
    o = np.lexsort((Z, col))
    level = np.empty(len(g), int)
    level[o] = np.tile(np.arange(nl), ncol)
    sline = np.linspace(0, L, nx)
    pts = np.c_[sline, np.full(nx, c0)] if axis == "x" else np.c_[np.full(nx, c0), sline]
    m0 = level == 0
    xy = np.empty((ncol, 2))
    xy[col[m0]] = np.c_[X[m0], Y[m0]]                 # each column's footprint position
    tri = Delaunay(xy)
    Zl, Tl = np.empty((nl, nx)), np.empty((nl, nx))
    for k in range(nl):
        m = level == k
        zz, tt = np.empty(ncol), np.empty(ncol)
        zz[col[m]], tt[col[m]] = Z[m], T[m]
        Zl[k] = LinearNDInterpolator(tri, zz)(pts)
        Tl[k] = LinearNDInterpolator(tri, tt)(pts)
    return sline, Zl, Tl


def main():
    args = sys.argv[1:]
    cfgf = arg(args, "--config", os.path.join(ROOT, "configs", "spread_ice.toml"))
    axis = arg(args, "--axis", "x")
    nx = arg(args, "--nx", 300, int)
    nz = arg(args, "--nz", 120, int)
    out = arg(args, "--out", None)
    if not args:
        raise SystemExit(__doc__)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    cfg = tomllib.load(open(cfgf, "rb"))
    c = cfg["case"]
    L, W = float(c["spread_length_km"]) * 1e3, float(c["width"])
    xc, yc = float(c.get("source_x_km", L / 2e3)) * 1e3, float(c.get("source_y_km", W / 2e3)) * 1e3
    P = eval_lua_parameters(os.path.join(ROOT, c.get("parameters", "case/parameters.lua")))
    Tsurf = float(cfg.get("parameter", {}).get("value", P["Tsurf"]))
    Q = float(c.get("arrhenius_Q", 60e3))
    Tref = float(c["arrhenius_Tref"]) if c.get("arrhenius_Tref") is not None else Tsurf
    T0 = float(P["T0"])
    eta_pas = float(P["viscosity"]) * 1e6 * 365.25 * 86400        # MPa a -> Pa s
    eta = lambda T: np.log10(eta_pas * np.exp(Q / 8.314 * (1 / np.clip(T, 150.0, T0) - 1 / Tref)))
    import plotly.graph_objects as go
    rows = []
    for p in args:
        s = load_state(p)
        t = float(s.meta.get("time", np.nan))
        sl, Zl, Tl = section(s, axis, yc if axis == "x" else xc, nx, L if axis == "x" else W)
        rows.append((t, sl, Zl, Tl, os.path.basename(p)))
        print(f"{os.path.basename(p)}: t {t:,.0f} yr, T on the section {np.nanmin(Tl):.2f} .. {np.nanmax(Tl):.2f} K, "
              f"surface {np.nanmin(Zl[-1]):.0f} .. {np.nanmax(Zl[-1]):.0f} m")
    rows.sort(key=lambda r: r[0])
    zmax = max(np.nanmax(r[2][-1]) for r in rows)
    zmin = min(np.nanmin(r[2][0]) for r in rows)
    zg = np.linspace(zmin, zmax, nz)
    lo_e, hi_e = eta(T0), eta(Tsurf)
    traces = []
    for t, sl, Zl, Tl, nm in rows:
        G = np.full((nz, len(sl)), np.nan)
        for j in range(len(sl)):
            zc, tc = Zl[:, j], Tl[:, j]
            if np.all(np.isfinite(zc)) and np.all(np.diff(zc) > 0):
                m = (zg >= zc[0]) & (zg <= zc[-1])
                G[m, j] = np.interp(zg[m], zc, tc)
        traces.append(go.Heatmap(x=sl / 1e3, y=zg, z=G.astype(np.float32), colorscale="RdBu_r", zmin=Tsurf, zmax=T0,
                                 colorbar=dict(title="T (K)"), visible=False, name=f"T {t:,.0f}",
                                 hovertemplate="%{x:.0f} km, z %{y:.0f} m<br>T %{z:.2f} K<extra></extra>"))
        traces.append(go.Heatmap(x=sl / 1e3, y=zg, z=eta(G).astype(np.float32), colorscale="Viridis_r", zmin=lo_e,
                                 zmax=hi_e, colorbar=dict(title="log10 eta (Pa s)"), visible=False,
                                 name=f"eta {t:,.0f}",
                                 hovertemplate="%{x:.0f} km, z %{y:.0f} m<br>log10 eta %{z:.2f}<extra></extra>"))
        traces.append(go.Scatter(x=sl / 1e3, y=Zl[-1], mode="lines", line=dict(color="black", width=1.2),
                                 visible=False, name="surface", hoverinfo="skip", showlegend=False))
    n = len(rows)
    for j in (3 * (n - 1), 3 * (n - 1) + 2):
        traces[j].visible = True
    fig = go.Figure(traces)
    import json
    titles = [f"section along {axis} through the source, t = {r[0]:,.0f} yr" for r in rows]
    steps = [dict(label=f"{r[0]:,.0f}", method="skip", args=[None], value=str(i)) for i, r in enumerate(rows)]
    buttons = [dict(label=nm_, method="skip", args=[None]) for nm_ in ("temperature", "log10 viscosity")]
    fig.update_layout(template="plotly_white", height=650, title=titles[-1],
                      xaxis=dict(title=f"{axis} (km)"), yaxis=dict(title="z (m)", range=[zmin, zmax * 1.05]),
                      sliders=[dict(active=n - 1, currentvalue=dict(prefix="t (yr) = "), steps=steps, pad=dict(t=40))],
                      updatemenus=[dict(type="buttons", direction="right", buttons=buttons, x=0, y=1.08,
                                        xanchor="left", yanchor="bottom", showactive=True, active=0)])
    js = """
var gd = document.getElementById('{plot_id}');
var N = %d, TITLES = %s, cur = {s: N - 1, f: 0};
function show() {
  var v = [];
  for (var s = 0; s < N; s++) { v.push(s === cur.s && cur.f === 0); v.push(s === cur.s && cur.f === 1); v.push(s === cur.s); }
  Plotly.restyle(gd, {visible: v});
  Plotly.relayout(gd, {'title.text': TITLES[cur.s]});
}
gd.on('plotly_buttonclicked', function (e) { cur.f = e.active; show(); });
gd.on('plotly_sliderchange', function (e) { cur.s = parseInt(e.step.value); show(); });
""" % (n, json.dumps(titles))
    out = out or os.path.join(os.path.dirname(os.path.abspath(args[0])), f"spread_section_{axis}.html")
    fig.write_html(out, include_plotlyjs="cdn", post_script=js)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
