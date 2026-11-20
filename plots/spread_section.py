#!/usr/bin/env python3
"""The spreading test (configs/spread_*.toml): a vertical cross-section through the source centre (y = the source's
y, along x; --axis y for the section along y), three panels stacked, one plotly page with a slider over the states (in
model-time order):
  * log10 of the viscosity (Pa s), eta = viscosity * exp(Q/R (1/T - 1/Tref)) as the model has it (SpreadViscosity in
    elmer/ess_utils.lua: T capped to 150 .. Tmax), Q, Tref and Tmax ([case] arrhenius_Tmax, default T0) from the config,
    `viscosity` from parameters.lua;
  * the horizontal speed sqrt(u^2 + v^2) (m/yr, log10), from the state's diagnostics Velocity 1, 2;
  * the temperature T (K).
On the speed panel, streamlines of the flow in the section plane (matplotlib streamplot of (velocity along the line,
w)), by default in the FRAME OF THE MARGIN: the margin's outward speed (from the states' margin positions along the
line, H > spread_thickness + 10 m) taken off, so a roll at the nose (surface -> nose -> bed) shows; --frame lab: as
computed; --density (1.2); --no-streams.
Each extruded level of the mesh is interpolated onto the section line (linear on the footprint's triangles), then every
column onto a regular z grid (--nz levels): the colours are the model's values, nothing smoothed across levels.

The panels are drawn at a fixed vertical exaggeration --vex (z against the section's horizontal distance; default: a
tenth of the old single-panel page's stretch, from the data's z range, printed); --x XMIN XMAX (km) to show only part of the line (the panels then get taller
at the same exaggeration); --panel PX: the window instead centred on the vent and just wide enough for panels PX
tall (--vex default 1 then: true aspect, cropped). Default window: where the surface stands > 10 m above the background layer in any state,
padded. Colour ranges: the data's (all states), not the model's possible range.

    python3 plots/spread_section.py STATE.npz [STATE2.npz ...] [--config configs/spread_vent.toml] [--axis x|y]
                                    [--vex V] [--x XMIN XMAX | --panel PX]
                                    [--frame front|lab] [--density 1.2] [--no-streams] [--nx 400] [--nz 120] [--width 1200] [--out FILE.html]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.case import eval_lua_parameters  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402


def arg(args, name, default, conv=str, n=1):
    if name in args:
        k = args.index(name)
        v = [conv(a) for a in args[k + 1:k + 1 + n]]
        del args[k:k + 1 + n]
        return v[0] if n == 1 else v
    return default


def section(s, axis, c0, sline):
    """(Z, T, horizontal speed, velocity along the line, vertical velocity) on the extruded levels (nl x len(sline)) of
    the state s along the line {y = c0} (axis x) or {x = c0} (axis y); the velocities NaN when the state has none"""
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator
    allc = {**s.diag, **s.frozen}
    cx, cy, cz = (allc[find_name(allc, f"coordinate {k}")] for k in (1, 2, 3))
    blk = {b.name.lower(): b for b in s.layout.blocks}
    bT = blk["temperature"]
    g = bT.gids

    def look(c):
        if np.all(np.diff(c[0]) > 0):
            i = np.clip(np.searchsorted(c[0], g), 0, len(c[0]) - 1)
            v = c[1][i].astype(float)
            v[c[0][i] != g] = np.nan
            return v
        d = dict(zip(c[0].tolist(), c[1]))
        return np.array([d.get(k, np.nan) for k in g.tolist()], float)

    X, Y, Z = look(cx), look(cy), look(cz)
    T = s.x[s.layout.slices()[bT.name]]
    try:
        V = [look(allc[find_name(allc, f"velocity {k}")]) for k in (1, 2, 3)]
        U = np.hypot(V[0], V[1])                                   # horizontal
        Ua, W = (V[0] if axis == "x" else V[1]), V[2]              # along the line, vertical
    except KeyError:
        U = Ua = W = np.full(len(g), np.nan)
    col = np.unique(np.c_[np.round(X, 1), np.round(Y, 1)], axis=0, return_inverse=True)[1].ravel()
    ncol = col.max() + 1
    nl = len(g) // ncol
    o = np.lexsort((Z, col))
    level = np.empty(len(g), int)
    level[o] = np.tile(np.arange(nl), ncol)
    nx = len(sline)
    pts = np.c_[sline, np.full(nx, c0)] if axis == "x" else np.c_[np.full(nx, c0), sline]
    m0 = level == 0
    xy = np.empty((ncol, 2))
    xy[col[m0]] = np.c_[X[m0], Y[m0]]                 # each column's footprint position
    tri = Delaunay(xy)
    out = [np.empty((nl, nx)) for _ in range(5)]
    for k in range(nl):
        m = level == k
        for f, v in zip(out, (Z, T, U, Ua, W)):
            vv = np.empty(ncol)
            vv[col[m]] = v[m]
            f[k] = LinearNDInterpolator(tri, vv)(pts)
    return out


def dome_extent(s, axis, c0, dh=10.0, band=5e3):
    """(lo, hi) along the line of where the surface stands more than dh above the background layer's (its median),
    from the Zs nodes within band of the line; None if nowhere"""
    allc = {**s.diag, **s.frozen}
    blk = {b.name.lower(): b for b in s.layout.blocks}
    g = blk["zs"].gids
    zs = s.layout.split(s.x)[blk["zs"].name]
    pos = []
    for k in (1, 2):
        c = allc[find_name(allc, f"coordinate {k}")]
        d = dict(zip(c[0].tolist(), c[1]))
        pos.append(np.array([d.get(i, np.nan) for i in g.tolist()], float))
    along, across = (pos[0], pos[1]) if axis == "x" else (pos[1], pos[0])
    m = (np.abs(across - c0) < band) & (zs > np.median(zs) + dh)
    return (float(along[m].min()), float(along[m].max())) if m.any() else None


def on_grid(zg, Zl, F):
    """the level field F (nl x nx) on the regular z grid zg (columns: linear between the levels; NaN outside the ice)"""
    G = np.full((len(zg), Zl.shape[1]), np.nan)
    for j in range(Zl.shape[1]):
        zc, fc = Zl[:, j], F[:, j]
        if np.all(np.isfinite(zc)) and np.all(np.diff(zc) > 0) and np.all(np.isfinite(fc)):
            m = (zg >= zc[0]) & (zg <= zc[-1])
            G[m, j] = np.interp(zg[m], zc, fc)
    return G


def main():
    args = sys.argv[1:]
    cfgf = arg(args, "--config", os.path.join(ROOT, "configs", "spread_vent.toml"))
    axis = arg(args, "--axis", "x")
    nx = arg(args, "--nx", 400, int)
    nz = arg(args, "--nz", 120, int)
    vex = arg(args, "--vex", None, float)
    width = arg(args, "--width", 1200, int)
    xwin = arg(args, "--x", None, float, n=2)
    panel_px = arg(args, "--panel", None, int)     # with it: the window cropped (about the vent) to panels this tall
    out = arg(args, "--out", None)
    frame = arg(args, "--frame", "front")           # the streamlines' frame: "front" (moving with the margin) or "lab"
    density = arg(args, "--density", 1.2, float)    # the streamlines' density (matplotlib streamplot)
    nostreams = "--no-streams" in args
    args = [a_ for a_ in args if a_ != "--no-streams"]
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
    Tmax = float(c["arrhenius_Tmax"]) if c.get("arrhenius_Tmax") is not None else T0
    eta_pas = float(P["viscosity"]) * 1e6 * 365.25 * 86400        # MPa a -> Pa s
    eta = lambda T: np.log10(eta_pas * np.exp(Q / 8.314 * (1 / np.clip(T, 150.0, Tmax) - 1 / Tref)))
    Lline = L if axis == "x" else W
    c0 = yc if axis == "x" else xc
    if xwin:
        a, b = xwin[0] * 1e3, xwin[1] * 1e3
    elif panel_px:
        # the window about the vent wide enough that the z range fills panel_px at exaggeration vex (default 1: true
        # aspect): width = (plot width px) x (z range) x vex / panel_px
        if vex is None:
            vex = 1.0
        zt, zb = -np.inf, np.inf
        for p in args:
            s_ = load_state(p)
            blk = {b_.name.lower(): b_ for b_ in s_.layout.blocks}
            zt = max(zt, float(np.nanmax(s_.layout.split(s_.x)[blk["zs"].name])))
            bv = s_.frozen[find_name(s_.frozen, "bedrock")][1]
            zb = min(zb, float(np.nanmin(bv)))
        wdt = (width - 200) * (1.05 * zt - zb) * vex / panel_px
        ctr = xc if axis == "x" else yc
        a, b = max(0.0, ctr - wdt / 2), min(Lline, ctr + wdt / 2)
        print(f"section window {a / 1e3:.2f} .. {b / 1e3:.2f} km: panels ~{panel_px} px at exaggeration {vex:g}")
    else:                                   # the flow's extent over all states, padded
        ext = [e for e in (dome_extent(load_state(p), axis, c0) for p in args) if e]
        if ext:
            lo, hi = min(e[0] for e in ext), max(e[1] for e in ext)
            pad = max(0.25 * (hi - lo), 20e3)
            a, b = max(0.0, lo - pad), min(Lline, hi + pad)
        else:
            a, b = 0.0, Lline
        print(f"section window {a / 1e3:.0f} .. {b / 1e3:.0f} km (--x to set)")
    sline = np.linspace(a, b, nx)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = []
    for p in args:
        s = load_state(p)
        t = float(s.meta.get("time", np.nan))
        Zl, Tl, Ul, Al, Wl = section(s, axis, c0, sline)
        rows.append((t, Zl, Tl, Ul, Al, Wl))
        print(f"{os.path.basename(p)}: t {t:,.0f} yr, T {np.nanmin(Tl):.2f} .. {np.nanmax(Tl):.2f} K, horizontal speed max "
              f"{np.nanmax(Ul) if np.isfinite(Ul).any() else np.nan:.3g} m/yr, surface {np.nanmin(Zl[-1]):.0f} .. "
              f"{np.nanmax(Zl[-1]):.0f} m")
    rows.sort(key=lambda r: r[0])
    # the margin on each side of the centre along the line (thickness above the layer > H0 + 10 m) and its speed
    H0l = float(c.get("spread_thickness", 100.0))
    ctr = xc if axis == "x" else yc
    Rf = []
    for r_ in rows:
        Hs = r_[1][-1] - r_[1][0]
        ins = np.where(np.isfinite(Hs) & (Hs > H0l + 10.0))[0]
        Rf.append(0.5 * (sline[ins[-1]] - sline[ins[0]]) if len(ins) else np.nan)
    Rf = np.array(Rf)
    tt = np.array([r_[0] for r_ in rows])
    Rdot = np.zeros(len(rows))
    if len(rows) > 1:
        Rdot = np.gradient(Rf, tt) if np.all(np.diff(tt) > 0) else Rdot
        Rdot = np.where(np.isfinite(Rdot), Rdot, 0.0)
    if frame == "lab":
        Rdot[:] = 0.0
    zmax = max(np.nanmax(r[1][-1]) for r in rows) * 1.05
    zmin = min(np.nanmin(r[1][0]) for r in rows)
    zg = np.linspace(zmin, zmax, nz)
    allU = np.concatenate([r[3][np.isfinite(r[3]) & (r[3] > 0)] for r in rows] + [np.array([1e-3])])
    u_hi = float(np.log10(np.percentile(allU, 99.5)))
    u_lo = u_hi - 4.0
    # the colour ranges from the data (all states; 0.5 .. 99.5 percentiles), not the model's whole possible range
    allT = np.concatenate([r[2][np.isfinite(r[2])].ravel() for r in rows])
    t_lo, t_hi = (float(v) for v in np.percentile(allT, [0.5, 99.5]))
    if t_hi - t_lo < 1.0:
        t_lo, t_hi = t_lo - 0.5, t_hi + 0.5
    plot_w = width - 200
    if vex is None:
        # a tenth of the old single-panel page's stretch: the whole line over ~plot_w px, the z range over ~470 px
        v_old = (Lline / plot_w) / ((zmax - zmin) / 470.0)
        vex = float(f"{v_old / 10:.2g}")
    # the panel height at exaggeration vex: the plot area ~ (width - margins) px for (b - a) of horizontal distance
    panel = max(40, int(round(plot_w * (zmax - zmin) * vex / (b - a))))
    gap = 70
    height = 3 * panel + 2 * gap + 230
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=gap / (height - 230),
                        subplot_titles=("log10 viscosity (Pa s)", "log10 horizontal speed (m/yr)", "temperature (K)"))
    cbar = lambda r, title: dict(title=dict(text=title, side="right"), lenmode="pixels", len=max(panel, 60),
                                 y=1 - ((r - 1) * (panel + gap) + panel / 2) / (height - 230), yanchor="middle",
                                 thickness=14, nticks=4)
    xs = sline / 1e3
    n = len(rows)
    streams = []
    if not nostreams:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    for i, (t, Zl, Tl, Ul, Al, Wl) in enumerate(rows):
        vis = i == n - 1
        # the streamlines in the section plane: (velocity along the line - the margin's speed outward, w), drawn on
        # the speed panel; in the front's frame the roll at the nose shows (surface -> nose -> bed)
        sx = sy = None
        if not nostreams and np.isfinite(Al).any():
            Ga, Gw = on_grid(zg, Zl, Al), on_grid(zg, Zl, Wl)
            urel = Ga - np.sign(sline - ctr)[None, :] * Rdot[i]
            msk = ~(np.isfinite(urel) & np.isfinite(Gw))
            fig_, ax_ = plt.subplots()
            try:
                sp = ax_.streamplot(xs, zg, np.ma.array(np.where(msk, 0, urel) / 1e3, mask=msk),
                                    np.ma.array(np.where(msk, 0, Gw), mask=msk), density=density)
                segs = sp.lines.get_segments()
                if segs:
                    sx = np.concatenate([np.r_[sg[:, 0], np.nan] for sg in segs])
                    sy = np.concatenate([np.r_[sg[:, 1], np.nan] for sg in segs])
            except Exception as e:                       # noqa: BLE001
                print(f"t {t:,.0f}: no streamlines ({e})")
            plt.close(fig_)
        G = on_grid(zg, Zl, Tl)
        with np.errstate(divide="ignore", invalid="ignore"):
            Gu = np.log10(on_grid(zg, Zl, Ul))
        for r, z_, cs, lo, hi, title, hv in (
                (1, eta(G), "Viridis_r", eta(t_hi), eta(t_lo), "log10 eta", "log10 eta %{z:.2f}"),
                (2, Gu, "Inferno", u_lo, u_hi, "log10 |u_h|", "log10 |u_h| %{z:.2f}"),
                (3, G, "RdBu_r", t_lo, t_hi, "T (K)", "T %{z:.2f} K")):
            fig.add_trace(go.Heatmap(x=xs, y=zg, z=np.where(np.isfinite(G), z_, np.nan).astype(np.float32),
                                     colorscale=cs, zmin=lo, zmax=hi, colorbar=cbar(r, title), visible=vis,
                                     hovertemplate=f"%{{x:.1f}} km, z %{{y:.0f}} m<br>{hv}<extra></extra>"),
                          row=r, col=1)
            fig.add_trace(go.Scatter(x=xs, y=Zl[-1], mode="lines", line=dict(color="black", width=1), visible=vis,
                                     hoverinfo="skip", showlegend=False), row=r, col=1)
        fig.add_trace(go.Scatter(x=sx if sx is not None else [], y=sy if sy is not None else [], mode="lines",
                                 line=dict(color="rgba(80,200,255,0.9)", width=1), visible=vis, hoverinfo="skip",
                                 showlegend=False), row=2, col=1)
        streams.append(Rdot[i])
    per = 7
    steps = [dict(label=f"{r[0]:,.0f}", method="update",
                  args=[{"visible": [j // per == i for j in range(per * n)]},
                        {"title.text": f"section along {axis} through the source, t = {r[0]:,.0f} yr "
                                       f"(vertical exaggeration {vex:g}; streamlines in the {frame} frame"
                                       + (f", margin {Rdot[i]:.3g} m/yr" if frame == "front" else "") + ")"}])
             for i, r in enumerate(rows)]
    for r in (1, 2, 3):
        fig.update_yaxes(title_text="z (m)", range=[zmin, zmax], scaleanchor=f"x{'' if r == 1 else r}",
                         scaleratio=vex / 1e3, constrain="domain", row=r, col=1)
        fig.update_xaxes(range=[a / 1e3, b / 1e3], constrain="domain", row=r, col=1)
    fig.update_xaxes(title_text=f"{axis} (km)", row=3, col=1)
    fig.update_layout(template="plotly_white", width=width, height=height, margin=dict(l=80, r=120, t=80, b=150),
                      title=f"section along {axis} through the source, t = {rows[-1][0]:,.0f} yr "
                            f"(vertical exaggeration {vex:g}; streamlines in the {frame} frame"
                            + (f", margin {Rdot[-1]:.3g} m/yr" if frame == "front" else "") + ")",
                      sliders=[dict(active=n - 1, currentvalue=dict(prefix="t (yr) = "), steps=steps, pad=dict(t=50))])
    out = out or os.path.join(os.path.dirname(os.path.abspath(args[0])), f"spread_section_{axis}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out} (panels {panel} px at vertical exaggeration {vex:g})")


if __name__ == "__main__":
    main()
