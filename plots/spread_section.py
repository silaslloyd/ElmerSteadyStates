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
On every panel, --tracers N (120; 0: none) markers placed along the drop's surface in one state (--seed-time T; default
the first state with a velocity, i.e. not a remeshed start) (where it is
thicker than the layer by 10 m), at --tracer-depths (m below the surface, default 0,10,20: a "crust"), and advected
through the states (in the section plane, axisymmetric: the velocity along the line and w, linear in time
between consecutive states, RK2), coloured by their starting position: the surface material's fate (e.g. rolled
under at the nose).
Each extruded level of the mesh is interpolated onto the section line (linear on the footprint's triangles), then every
column onto a regular z grid (--nz levels): the colours are the model's values, nothing smoothed across levels.

The panels are drawn at a fixed vertical exaggeration --vex (z against the section's horizontal distance; default: a
tenth of the old single-panel page's stretch, from the data's z range, printed); --x XMIN XMAX (km) to show only part of the line (the panels then get taller
at the same exaggeration); --panel PX: the window instead centred on the vent and just wide enough for panels PX
tall (--vex default 1 then: true aspect, cropped). Default window: where the surface stands > 10 m above the background layer in any state,
padded. --height H: the exaggeration instead chosen so the page is H px tall (the panels fill it). Colour ranges: the data's (all states), not the model's possible range.

    python3 plots/spread_section.py STATE.npz [STATE2.npz ...] [--config configs/spread_vent.toml] [--axis x|y]
                                    [--vex V | --height H] [--x XMIN XMAX | --panel PX]
                                    [--frame front|lab] [--density 1.2] [--no-streams] [--tracers 120] [--tracer-depths 0,10,20] [--seed-time T] [--nx 400] [--nz 120] [--width 1200] [--out FILE.html]
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


def fmt_t(t):
    """a model time (yr) for labels, in a unit that shows it: yr, days, hours or minutes"""
    if t == 0:
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

    def look_g(c, gg):
        """c's values at the node ids gg (NaN where missing)"""
        o2 = np.argsort(c[0])
        i = np.clip(np.searchsorted(c[0][o2], gg), 0, len(c[0]) - 1)
        v = c[1][o2][i].astype(float)
        v[c[0][o2][i] != gg] = np.nan
        return v
    T = s.x[s.layout.slices()[bT.name]]
    try:
        V = [look(allc[find_name(allc, f"velocity {k}")]) for k in (1, 2, 3)]
        U = np.hypot(V[0], V[1])                                   # horizontal
        Ua, W = (V[0] if axis == "x" else V[1]), V[2]              # along the line, vertical
    except KeyError:
        U = Ua = W = np.full(len(g), np.nan)
    keys, col = np.unique(np.c_[np.round(X, 1), np.round(Y, 1)], axis=0, return_inverse=True)
    col = col.ravel()
    ncol = col.max() + 1
    nl = len(g) // ncol
    o = np.lexsort((Z, col))
    level = np.empty(len(g), int)
    level[o] = np.tile(np.arange(nl), ncol)
    # the node heights from the state's own Zs: a state whose stored coordinates are not its geometry (a start state
    # made from another, e.g. the drop added to the flat layer's Zs, jobs/spread_drop.py) is drawn as it is. Each
    # column's levels keep their relative positions zeta between its bed (level 0) and its top.
    bz = blk.get("zs")
    if bz is not None:
        zs = s.layout.split(s.x)[bz.name]
        xz, yz = look_g(cx, bz.gids), look_g(cy, bz.gids)
        # each Zs node's column: the same rounding as the columns' keys, matched by a sorted search
        kz = np.c_[np.round(xz, 1), np.round(yz, 1)]
        kk = keys[:, 0] * 1e7 + keys[:, 1]
        qq = kz[:, 0] * 1e7 + kz[:, 1]
        o_ = np.argsort(kk)
        pos = np.clip(np.searchsorted(kk[o_], qq), 0, len(kk) - 1)
        hit = kk[o_][pos] == qq
        zcol = np.full(ncol, np.nan)
        zcol[o_[pos[hit]]] = zs[hit]
        bot = np.empty(ncol)
        top = np.empty(ncol)
        bot[col[level == 0]] = Z[level == 0]
        top[col[level == nl - 1]] = Z[level == nl - 1]
        ok = np.isfinite(zcol)
        zeta = (Z - bot[col]) / np.where(top[col] > bot[col], top[col] - bot[col], 1.0)
        Z = np.where(ok[col], bot[col] + zeta * (zcol[col] - bot[col]), Z)
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


def col_vel(row, sline, xp, zp):
    """the velocity (along the line, vertical) of a state's section `row` (t, Z, T, U, A, W levels) at the points
    (xp, zp) (m): linear between the two neighbouring columns, along each column linear in z between the levels;
    NaN outside the line; z clamped to the column"""
    _, Zl, _, _, Al, Wl = row
    j = np.clip(np.searchsorted(sline, xp) - 1, 0, len(sline) - 2)
    f = np.clip((xp - sline[j]) / (sline[j + 1] - sline[j]), 0, 1)
    ua, wa = np.full(len(xp), np.nan), np.full(len(xp), np.nan)
    for i in range(len(xp)):
        if not (sline[0] <= xp[i] <= sline[-1]):
            continue
        Zc = (1 - f[i]) * Zl[:, j[i]] + f[i] * Zl[:, j[i] + 1]
        Ac = (1 - f[i]) * Al[:, j[i]] + f[i] * Al[:, j[i] + 1]
        Wc = (1 - f[i]) * Wl[:, j[i]] + f[i] * Wl[:, j[i] + 1]
        if not (np.all(np.isfinite(Zc)) and np.all(np.isfinite(Ac)) and np.all(np.isfinite(Wc))):
            continue
        z = min(max(zp[i], Zc[0]), Zc[-1])
        ua[i], wa[i] = np.interp(z, Zc, Ac), np.interp(z, Zc, Wc)
    return ua, wa


def surf_bed(row, sline, xp):
    """the surface and the bed of a state's section at xp"""
    return np.interp(xp, sline, row[1][-1]), np.interp(xp, sline, row[1][0])


def track(rows, sline, x0, z0, nsub=20, on_surface=None, Hrel=None):
    """tracers advected in the section plane (axisymmetric: the velocity along the line and w) through the states'
    times: RK2 with the velocity linear in time between consecutive states, nsub substeps per interval; kept between
    the bed and the surface. on_surface (bool per tracer): tracers ON the free surface (a material surface: nothing
    added or removed there) stay on it, moved along it by the surface velocity, until they reach the front (where the
    thickness falls below Hrel, the layer + the margin's dh); there they leave it (rolled under) and move with the flow.
    Returns [(x, z) at each state's time]"""
    x, z = np.asarray(x0, float).copy(), np.asarray(z0, float).copy()
    pin = np.zeros(len(x), bool) if on_surface is None else np.asarray(on_surface, bool).copy()
    out = [(x.copy(), z.copy())]
    for i in range(len(rows) - 1):
        r0, r1 = rows[i], rows[i + 1]
        t0, t1 = r0[0], r1[0]
        h = (t1 - t0) / nsub
        v0 = lambda xp, zp: col_vel(r0, sline, xp, zp)
        v1 = lambda xp, zp: col_vel(r1, sline, xp, zp)

        def vel(xp, zp, a):
            ua0, wa0 = v0(xp, zp)
            ua1, wa1 = v1(xp, zp)
            return (1 - a) * ua0 + a * ua1, (1 - a) * wa0 + a * wa1
        for k in range(nsub):
            a = k / nsub
            u1, w1 = vel(x, z, a)
            xm, zm = x + 0.5 * h * u1, z + 0.5 * h * w1
            u2, w2 = vel(xm, zm, a + 0.5 / nsub)
            x, z = x + h * u2, z + h * w2
            a2 = (k + 1) / nsub
            s0, b0 = surf_bed(r0, sline, x)
            s1, b1 = surf_bed(r1, sline, x)
            zs, zb = (1 - a2) * s0 + a2 * s1, (1 - a2) * b0 + a2 * b1
            if pin.any():
                # on the surface: kept on it; released where the thickness falls below Hrel (the front)
                front = (zs - zb) < (Hrel if Hrel is not None else 0.0)
                rel = pin & front
                pin = pin & ~front
                z = np.where(pin, zs, z)
                z = np.where(rel, np.minimum(z, zs), z)
            z = np.minimum(np.maximum(z, zb), zs)
        out.append((x.copy(), z.copy()))
    return out


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
    page_h = arg(args, "--height", None, int)      # with it: the exaggeration set so the three panels fill a page this tall
    xwin = arg(args, "--x", None, float, n=2)
    panel_px = arg(args, "--panel", None, int)     # with it: the window cropped (about the vent) to panels this tall
    out = arg(args, "--out", None)
    frame = arg(args, "--frame", "front")           # the streamlines' frame: "front" (moving with the margin) or "lab"
    density = arg(args, "--density", 1.2, float)    # the streamlines' density (matplotlib streamplot)
    nostreams = "--no-streams" in args
    ntr = arg(args, "--tracers", 120, int)          # tracers along the drop's first surface (0: none)
    tdep = [float(v) for v in arg(args, "--tracer-depths", "0,10,20").split(",")]   # m below the surface: the "crust"
    tseed = arg(args, "--seed-time", None, float)
    pin_surface = "--pin-surface" in args            # the surface tracers held on the surface (off: free advection)
    args = [a_ for a_ in args if a_ != "--pin-surface"]   # the state the tracers start on (default: the first with a velocity)
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
    cap = float(c.get("viscosity_cap", 0.0))            # SpreadViscosity's cap: at most viscosity x cap
    eta = lambda T: np.minimum(np.log10(eta_pas * np.exp(Q / 8.314 * (1 / np.clip(T, 150.0, Tmax) - 1 / Tref))),
                               np.log10(eta_pas * cap) if cap > 0 else np.inf)
    # the stiff base layer (SpreadViscosity(T, z)): x f^s, s = (1 - tanh((z - zb - centre) / d)) / 2
    lay_f = float(c.get("layer_viscosity_factor", 1.0))
    lay_c = float(c.get("layer_step_centre", c.get("layer_thickness", c.get("spread_thickness", 100.0))))
    lay_d = float(c.get("layer_smooth", 20.0))
    zb0 = float(c.get("spread_bed", 0.0))

    def layer_log10(zgrid):
        """log10 of the layer factor at the heights zgrid (any shape)"""
        if lay_f == 1.0:
            return np.zeros_like(zgrid)
        u = np.clip((zgrid - zb0 - lay_c) / lay_d, -20, 20)
        return np.log10(lay_f) * 0.5 * (1 - np.tanh(u))
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
    gap = 70
    if page_h is not None:
        # the exaggeration that makes the page page_h px tall: 3 panels + 2 gaps + 230 px of title, axis and slider
        pnl = max(40, (page_h - 230 - 2 * gap) // 3)
        vex = float(f"{pnl * (b - a) / (plot_w * (zmax - zmin)):.2g}")
    if vex is None:
        # a tenth of the old single-panel page's stretch: the whole line over ~plot_w px, the z range over ~470 px
        v_old = (Lline / plot_w) / ((zmax - zmin) / 470.0)
        vex = float(f"{v_old / 10:.2g}")
    # the panel height at exaggeration vex: the plot area ~ (width - margins) px for (b - a) of horizontal distance
    panel = max(40, int(round(plot_w * (zmax - zmin) * vex / (b - a))))
    height = 3 * panel + 2 * gap + 230
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=gap / (height - 230),
                        subplot_titles=("log10 viscosity (Pa s)", "log10 horizontal speed (m/yr), streamlines",
                                        "temperature (K)"))
    cbar = lambda r, title: dict(title=dict(text=title, side="right"), lenmode="pixels", len=max(panel, 60),
                                 y=1 - ((r - 1) * (panel + gap) + panel / 2) / (height - 230), yanchor="middle",
                                 thickness=14, nticks=4)
    xs = sline / 1e3
    n = len(rows)
    # tracers: a layer on the first state's surface, advected through the states (the third panel)
    trk = None
    hasv = [bool(np.isfinite(r_[4]).any()) for r_ in rows]
    if ntr > 0 and any(hasv):
        # seeded on the surface of one state (default the first with a velocity: a remeshed start has none, and the
        # first interval from it is where the drop changes fastest), tracked from there on; before it, none
        if tseed is None:
            i0 = hasv.index(True)
        else:
            i0 = int(np.argmin([abs(r_[0] - tseed) for r_ in rows]))
        rv = list(rows)
        for i in range(len(rows) - 1, -1, -1):          # a later state without the velocity: the nearest one's
            if not hasv[i]:
                j = next((k for k in range(i + 1, len(rows)) if hasv[k]), None)
                if j is None:
                    j = max(k for k in range(i) if hasv[k])
                rv[i] = rows[i][:4] + (rows[j][4], rows[j][5])
        Hs0 = rows[i0][1][-1] - rows[i0][1][0]
        ins = np.where(np.isfinite(Hs0) & (Hs0 > float(c.get("spread_thickness", 100.0)) + 10.0))[0]
        xa, xb = (sline[ins[0]], sline[ins[-1]]) if len(ins) else (sline[0], sline[-1])
        xs0 = np.linspace(xa, xb, ntr)
        zs0 = np.interp(xs0, sline, rows[i0][1][-1])
        x0 = np.concatenate([xs0 for _ in tdep])
        z0 = np.concatenate([zs0 - d_ for d_ in tdep])
        onsurf = (np.concatenate([np.full(len(xs0), d_ == 0.0) for d_ in tdep]) if pin_surface else None)
        part = track(rv[i0:], sline, x0, z0, on_surface=onsurf,
                     Hrel=float(c.get("spread_thickness", 100.0)) + float(c.get("margin_dh", 10.0)))
        trk = [None] * i0 + part
        print(f"tracers: {ntr} x {len(tdep)} on the surface at t {rows[i0][0]:,.0f} yr, advected to "
              f"t {rows[-1][0]:,.0f} yr")
    elif ntr > 0:
        print("tracers: none (no state has the velocity)")
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
        ax_x, ax_y, ax_a = [], [], []                     # the arrowheads: one per streamline, at its middle
        if not nostreams and np.isfinite(Al).any():
            Ga, Gw = on_grid(zg, Zl, Al), on_grid(zg, Zl, Wl)
            urel = Ga - np.sign(sline - ctr)[None, :] * Rdot[i]
            msk = ~(np.isfinite(urel) & np.isfinite(Gw))
            fig_, ax_ = plt.subplots()
            try:
                sp = ax_.streamplot(xs, zg, np.ma.array(np.where(msk, 0, urel) / 1e3, mask=msk),
                                    np.ma.array(np.where(msk, 0, Gw), mask=msk), density=density)
                segs = sp.lines.get_segments()        # one polyline per streamline, in the flow direction
                if segs:
                    sx = np.concatenate([np.r_[sg[:, 0], np.nan] for sg in segs])
                    sy = np.concatenate([np.r_[sg[:, 1], np.nan] for sg in segs])
                for sg in segs:
                    if len(sg) < 3:
                        continue
                    # the middle by length on the screen (x km -> m, z m x vex), the direction there
                    d = np.hypot(np.diff(sg[:, 0]) * 1e3, np.diff(sg[:, 1]) * vex)
                    cl = np.r_[0.0, np.cumsum(d)]
                    if cl[-1] <= 0:
                        continue
                    j = int(np.clip(np.searchsorted(cl, cl[-1] / 2), 1, len(sg) - 1))
                    dx, dz = (sg[j, 0] - sg[j - 1, 0]) * 1e3, (sg[j, 1] - sg[j - 1, 1]) * vex
                    ax_x.append(sg[j, 0])
                    ax_y.append(sg[j, 1])
                    ax_a.append(float(np.degrees(np.arctan2(dx, dz))))   # plotly: clockwise from up
            except Exception as e:                       # noqa: BLE001
                print(f"t {t:,.0f}: no streamlines ({e})")
            plt.close(fig_)
        G = on_grid(zg, Zl, Tl)
        with np.errstate(divide="ignore", invalid="ignore"):
            Gu = np.log10(on_grid(zg, Zl, Ul))
        for r, z_, cs, lo, hi, title, hv in (
                (1, eta(G) + layer_log10(np.broadcast_to(zg[:, None], G.shape)), "Viridis_r", eta(t_hi),
                 eta(t_lo) + (np.log10(lay_f) if lay_f != 1.0 else 0.0), "log10 eta", "log10 eta %{z:.2f}"),
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
        fig.add_trace(go.Scatter(x=ax_x, y=ax_y, mode="markers", marker=dict(symbol="triangle-up", size=7, angle=ax_a,
                                 color="rgba(80,200,255,1)", line=dict(width=0)), visible=vis, hoverinfo="skip",
                                 showlegend=False), row=2, col=1)
        have = trk is not None and trk[i] is not None
        tx, tz = (trk[i][0] / 1e3, trk[i][1]) if have else ([], [])
        tc = (np.tile(np.arange(ntr), len(tdep)) if have else [])         # coloured by the starting position
        for r in (1, 2, 3):
            fig.add_trace(go.Scatter(x=tx, y=tz, mode="markers",
                                     marker=dict(size=4, color=tc, colorscale="Turbo", line=dict(width=0.4, color="black")),
                                     visible=vis, hovertemplate="tracer %{marker.color}<br>%{x:.2f} km, z %{y:.0f} m"
                                                                "<extra></extra>", showlegend=False), row=r, col=1)
        streams.append(Rdot[i])
    per = 11
    steps = [dict(label=fmt_t(r[0]), method="update",
                  args=[{"visible": [j // per == i for j in range(per * n)]},
                        {"title.text": f"section along {axis} through the source, t = {fmt_t(r[0])} "
                                       f"(vertical exaggeration {vex:g}; streamlines in the {frame} frame"
                                       + (f", margin {Rdot[i]:.3g} m/yr" if frame == "front" else "") + ")"}])
             for i, r in enumerate(rows)]
    for r in (1, 2, 3):
        fig.update_yaxes(title_text="z (m)", range=[zmin, zmax], scaleanchor=f"x{'' if r == 1 else r}",
                         scaleratio=vex / 1e3, constrain="domain", row=r, col=1)
        fig.update_xaxes(range=[a / 1e3, b / 1e3], constrain="domain", row=r, col=1)
    fig.update_xaxes(title_text=f"{axis} (km)", row=3, col=1)
    fig.update_layout(template="plotly_white", width=width, height=height, margin=dict(l=80, r=120, t=80, b=150),
                      title=f"section along {axis} through the source, t = {fmt_t(rows[-1][0])} "
                            f"(vertical exaggeration {vex:g}; streamlines in the {frame} frame"
                            + (f", margin {Rdot[-1]:.3g} m/yr" if frame == "front" else "") + ")",
                      sliders=[dict(active=n - 1, currentvalue=dict(prefix="t = "), steps=steps, pad=dict(t=50))])
    out = out or os.path.join(os.path.dirname(os.path.abspath(args[0])), f"spread_section_{axis}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out} (panels {panel} px at vertical exaggeration {vex:g})")


if __name__ == "__main__":
    main()
