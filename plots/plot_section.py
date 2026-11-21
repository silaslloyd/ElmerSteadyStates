#!/usr/bin/env python3
"""A flowline state's temperature as a filled-contour section (plotly): x along the flow, z elevation, the ice
temperature on a regular z grid column by column (no smoothing across columns), the upper surface Zs, the ice base Zb
and the bed drawn in. The temperature is clipped to the surface temperature (the state's parameter, Tsurf; --tsurf K to set it) ..
melting (273.15 K); how many nodes were outside is printed. --no-clip: the temperature as it is, the colours over
its whole range (e.g. to check a limiter). The
flowline mesh is one element wide: the nodes at the smallest y are used.

    python3 plots/plot_section.py STATE.npz [STATE2.npz ...] [--x XMIN XMAX (km)] [--tmin K] [--tmax K] [--tsurf K] [--umax M/YR] [--nz 150] [--nx 2000]
                                  [--contour] [--no-clip] [--rock]
                                  [--out FILE.html]

Above the plot: a "colour field" toggle (off: the surface lines alone, no temperature / velocity colours), u max (the
speed's upper colour limit, typed in), u min (the lower limit on the log scale) and a "log u" toggle (log10 |u|, floored at 1e-3 m/yr). Buttons switch between the temperature and the velocities u (along the flow; colours up to the fastest grounded
ice, --umax to set them) and w (vertical, symmetric about 0), from the state's diagnostics Velocity 1 / Velocity 3.

A heatmap of the mesh's own columns (averaged along x only when there are more than --nx, default 2000) x --nz levels (the page stays fast); --contour: filled contours
instead (smoother, much slower in the browser).

--rock: the bedrock's temperature too (down to the bottom of the 10 zscale rock), not only the ice.

Several states (e.g. the fully temperate steady state and the bed sweep's points/state_*.npz): one page, a slider at
the bottom chooses the state (the sweep points ordered by d, from the longest temperate patch down; the patch head
in the title), the buttons the field; the colours are the same for every state. Default --nz 100 then
(the page holds every state); the page: the first state's name + _sections.html.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from plot_state import column_grid, load, on_side  # noqa: E402
from ess.statefile import find_name  # noqa: E402


def binned(x, z, v, nz, nx):
    """column_grid, then the columns averaged into at most nx bins along x (the page stays light)"""
    xg, zg, G = column_grid(x, z, v, nz)
    if len(xg) <= nx:
        return xg, zg, G.astype(np.float32)
    edges = np.linspace(xg[0], xg[-1], nx + 1)
    k = np.clip(np.searchsorted(edges, xg, side="right") - 1, 0, nx - 1)
    Gb = np.full((len(zg), nx), np.nan, np.float32)
    xb = np.full(nx, np.nan)
    with np.errstate(invalid="ignore"):
        for b in range(nx):
            m = k == b
            if m.any():
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    Gb[:, b] = np.nanmean(G[:, m], axis=1)
                xb[b] = xg[m].mean()
    ok = np.isfinite(xb)
    return xb[ok], zg, Gb[:, ok]


def field_trace(go, contour, x, z, G, lo, hi, cs, cbt, hov, name, visible=True):
    if contour:
        return go.Contour(x=x, y=z, z=G, colorscale=cs, zmin=lo, zmax=hi, visible=visible,
                          contours=dict(start=lo, end=hi, size=(hi - lo) / 30, coloring="fill", showlines=False),
                          colorbar=dict(title=cbt), connectgaps=False, hovertemplate=hov, name=name)
    return go.Heatmap(x=x, y=z, z=G, colorscale=cs, zmin=lo, zmax=hi, visible=visible, zsmooth=False,
                      colorbar=dict(title=cbt), hoverongaps=False, hovertemplate=hov, name=name)


def opt(args, name, n=1, conv=float, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = [conv(a) for a in args[k + 1:k + 1 + n]]
    return args[:k] + args[k + 1 + n:], (v[0] if n == 1 else v)


def state_meta(path):
    """the sweep's d (m) and grounding line (m) when the state is a sweep point, else None"""
    with np.load(path) as z:
        g = lambda k: float(z[k]) if k in z.files else None
        return g("meta::sweep_d"), g("meta::x_gl")


def iter_tag(path):
    """the label suffix of a Newton iterate (jobs/newton_iterates.py): ', Newton it k'; '' for other states"""
    with np.load(path) as z:
        return f", Newton it {int(z['meta::newton_it'])}" if "meta::newton_it" in z.files else ""


def build(go, path, a):
    """one state's traces: the fields (temperature, u, w: heatmaps) and the lines (Zs, Zb, bed);
    returns (fields, lines, field names, field ranges)"""
    xwin, nz, nx, contour, noclip, tsurf, rock = a["xwin"], a["nz"], a["nx"], a["contour"], a["noclip"], a["tsurf"], a["rock"]
    layout, fields, diag, frozen, title, _ = load(path, 0)
    coords = {f"coordinate {c}": diag[find_name(diag, f"coordinate {c}")] for c in (1, 2, 3)}
    gx, X = coords["coordinate 1"]
    gz, Z = coords["coordinate 3"]

    def xz(g):
        return X[np.searchsorted(gx, g)] / 1e3, Z[np.searchsorted(gz, g)]

    def profile(name, src=None):
        if src is None:
            nm = next((n for n in layout.names if n.lower() == name), None)
            if nm is None:
                return None
            g = next(b.gids for b in layout.blocks if b.name == nm)
            v = fields[nm]
        else:
            g, v = src
        s = on_side(g, coords)
        x_ = xz(g[s])[0]
        o = np.argsort(x_)
        return x_[o], v[s][o]
    pzs, pzb = profile("zs"), profile("zb")
    pbed = profile(None, frozen[find_name(frozen, "bedrock")]) if any(k.lower() == "bedrock" for k in frozen) else None
    if pbed is not None:                         # the bedrock record holds every node of a column: its base level
        xb_, vb_ = pbed
        k_ = np.unique(np.round(xb_, 6), return_index=True)[1]
        pbed = (xb_[k_], vb_[k_])
    tname = next(n for n in layout.names if n.lower() == "temperature")
    gT = next(b.gids for b in layout.blocks if b.name == tname)
    s = on_side(gT, coords)
    xt, zt = xz(gT[s])
    T = fields[tname][s]
    if not rock and pzb is not None:             # the ice only: at or above the base
        ice = zt >= np.interp(xt, *pzb) - 0.5
        xt, zt, T = xt[ice], zt[ice], T[ice]
    if xwin:
        m = (xt >= xwin[0]) & (xt <= xwin[1])
        xt, zt, T = xt[m], zt[m], T[m]
    # clipped at melting above and at the surface temperature below (the state's parameter p, Tsurf; --tsurf to
    # set it); how many nodes were outside is reported
    ts = tsurf if tsurf is not None else float(np.load(path)["p"])
    above, below = int(np.sum(T > 273.15 + 1e-6)), int(np.sum(T < ts - 1e-6))
    if not noclip:
        T = np.clip(T, ts, 273.15)
    xg, zg, G = binned(xt, zt, T, nz, nx)
    lo = a["tmin"] if a["tmin"] is not None else (float(np.nanmin(G)) if noclip else ts)
    hi = a["tmax"] if a["tmax"] is not None else (float(np.nanmax(G)) if noclip else 273.15)
    print(f"{path}: {len(xg)} columns, " + ("T NOT clipped (--no-clip)" if noclip else f"T clipped to {ts:.2f} .. 273.15 K") + f" ({above} plotted nodes were above melting, "
          f"{below} below the surface temperature); colours {lo:.2f} .. {hi:.2f}")
    ftr = [field_trace(go, contour, xg, zg, G, lo, hi, "RdBu_r", "T (K)",
                       "x %{x:.1f} km<br>z %{y:.0f} m<br>T %{z:.2f} K<extra></extra>", "T")]
    names, ranges = ["temperature"], [(lo, hi)]
    # the velocities (the state's diagnostics Velocity 1 / Velocity 3, on the ice nodes), as two more fields
    for c, lab, cs in ((1, "u (along-flow velocity, m/yr)", "Viridis"), (3, "w (vertical velocity, m/yr)", "RdBu_r")):
        try:
            gv, vv = diag[find_name(diag, f"velocity {c}")]
        except KeyError:
            print(f"no Velocity {c} in the state: no {lab.split()[0]} view")
            continue
        sv = on_side(gv, coords)
        xv, zv = xz(gv[sv])
        val = vv[sv]
        if pzb is not None:
            m = zv >= np.interp(xv, *pzb) - 0.5
            xv, zv, val = xv[m], zv[m], val[m]
        if xwin:
            m = (xv >= xwin[0]) & (xv <= xwin[1])
            xv, zv, val = xv[m], zv[m], val[m]
        xg_, zg_, G_ = binned(xv, zv, val, nz, nx)
        if c == 1:          # up to the fastest grounded ice (the shelf, much faster, saturates; --umax to set it)
            xgl = None
            if pzb is not None and pbed is not None:
                fl = pzb[1] - np.interp(pzb[0], *pbed) > 1.0          # the base above the bed: floating
                xgl = float(pzb[0][np.argmax(fl)]) if fl.any() else None
            gsel = (xg_ <= xgl) if xgl is not None else np.ones(len(xg_), bool)
            hi_ = a["umax"] if a["umax"] is not None else float(np.nanmax(G_[:, gsel]))
            lo_ = 0.0
            ugrid = (xg_, zg_, G_)
        else:
            hi_ = float(np.nanpercentile(np.abs(G_), 99)) or 1e-6
            lo_ = -hi_
        print(f"  {lab}: {np.nanmin(G_):.4g} .. {np.nanmax(G_):.4g}")
        ftr.append(field_trace(go, contour, xg_, zg_, G_, lo_, hi_, cs, lab.split()[0] + " (m/yr)",
                               "x %{x:.1f} km<br>z %{y:.0f} m<br>" + lab.split()[0] + " %{z:.4g} m/yr<extra></extra>",
                               lab, visible=False))
        names.append(lab.split(" (")[0])
        ranges.append((lo_, hi_))
    ulog_lo = None
    if "u" in names:            # log10 |u| (the page's "log u" toggle), floored at 1e-3 m/yr; after the fields
        xg_, zg_, G_ = ugrid
        with np.errstate(invalid="ignore", divide="ignore"):
            L_ = np.log10(np.maximum(np.abs(G_), 1e-3)).astype(np.float32)
        L_[~np.isfinite(G_)] = np.nan
        pos = np.abs(G_[np.isfinite(G_) & (np.abs(G_) > 1e-3)])
        ulog_lo = float(np.log10(np.percentile(pos, 1))) if pos.size else -3.0
        ftr.append(field_trace(go, contour, xg_, zg_, L_, ulog_lo, 0.0, "Viridis", "u (m/yr)",
                               "x %{x:.1f} km<br>z %{y:.0f} m<br>log10 u %{z:.3f}<extra></extra>",
                               "log10 u (along-flow velocity, m/yr)", visible=False))
    lines = []
    for p_, nm, col, dash in ((pzs, "Zs (surface)", "black", None), (pzb, "Zb (ice base)", "#8e44ad", None),
                              (pbed, "bed", "#7f6000", "dot")):
        if p_ is None:
            continue
        x_, z_ = p_
        if xwin:
            m = (x_ >= xwin[0]) & (x_ <= xwin[1])
            x_, z_ = x_[m], z_[m]
        lines.append(go.Scatter(x=x_, y=z_, mode="lines", name=nm, line=dict(color=col, width=1.6, dash=dash),
                                hovertemplate=f"x %{{x:.1f}} km<br>{nm} %{{y:.0f}} m<extra></extra>"))
    d, xgl_m = state_meta(path)                 # a sweep point: d, the temperate patch's length (the title)
    lab = (f"d = {d / 1e3:.1f} km (temperate from {(xgl_m - d) / 1e3:.1f} km){iter_tag(path)}"
           if d is not None and xgl_m is not None else os.path.basename(path))
    return ftr, lines, names, ranges, f"{lab}, {title}", ulog_lo


def main():
    args = sys.argv[1:]
    args, xwin = opt(args, "--x", n=2)
    args, tmin = opt(args, "--tmin")
    args, tmax = opt(args, "--tmax")
    args, nz = opt(args, "--nz", conv=int, default=None)
    args, nx = opt(args, "--nx", conv=int, default=2000)
    contour = "--contour" in args               # filled contours (slow in the browser) instead of a heatmap
    noclip = "--no-clip" in args                # the temperature as it is, the colours over its whole range
    args = [a for a in args if a not in ("--contour", "--no-clip")]
    args, tsurf = opt(args, "--tsurf")
    args, umax = opt(args, "--umax")
    args, out = opt(args, "--out", conv=str)
    rock = "--rock" in args
    args = [a for a in args if a != "--rock"]
    if not args:
        raise SystemExit(__doc__)
    import plotly.graph_objects as go
    # the states: the given order, sweep points by d from the longest patch (the most temperate) down
    from state_pick import prefer_fine
    paths = prefer_fine(args)               # one state per sweep length d: the finest mesh's
    meta = {p_: state_meta(p_)[0] for p_ in paths}
    plain = [p_ for p_ in paths if meta[p_] is None]
    swept = sorted((p_ for p_ in paths if meta[p_] is not None), key=lambda p_: -meta[p_])
    paths = plain + swept
    if nz is None:
        nz = 150 if len(paths) == 1 else 100          # several states: a lighter page
    a = dict(xwin=xwin, nz=nz, nx=nx, contour=contour, noclip=noclip, tsurf=tsurf, rock=rock, tmin=tmin, tmax=tmax,
             umax=umax)
    built = [build(go, p_, a) for p_ in paths]
    names = built[0][2]
    nf = len(names)
    if any(b[2] != names for b in built):
        raise SystemExit("the states hold different fields (velocities missing in some): plot them separately")
    # one colour range per field over all the states (the slider compares like with like)
    for j in range(nf):
        lo = min(b[3][j][0] for b in built)
        hi = max(b[3][j][1] for b in built)
        if names[j] == "w":
            hi = max(abs(lo), abs(hi))
            lo = -hi
        for b in built:
            b[0][j].update(zmin=lo, zmax=hi)
            if a["contour"]:
                b[0][j].update(contours=dict(start=lo, end=hi, size=(hi - lo) / 30))
        print(f"{names[j]}: colours {lo:.4g} .. {hi:.4g} (all states)")
    uj = names.index("u") if "u" in names else -1
    umax0 = max(b[3][uj][1] for b in built) if uj >= 0 else 1.0
    umin0 = 10 ** min(b[5] for b in built) if uj >= 0 else 1e-3
    if uj >= 0:                 # the log u traces: the same upper limit, ticks at the powers of ten
        lo_l, hi_l = np.log10(umin0), np.log10(umax0)
        tv = list(range(int(np.floor(lo_l)), int(np.ceil(hi_l)) + 1))
        for b in built:
            b[0][nf].update(zmin=lo_l, zmax=hi_l, colorbar=dict(title="u (m/yr)", tickvals=tv,
                                                                ticktext=[f"{10.0 ** v:g}" for v in tv]))
    traces, per = [], []
    for k, (ftr, lines, _, _, _, _) in enumerate(built):
        for j, t in enumerate(ftr):
            t.update(visible=(k == 0 and j == 0))
        for t in lines:
            t.update(visible=(k == 0))
        traces += ftr + lines
        per.append(len(ftr) + len(lines))
    fig = go.Figure(traces)
    # the axes fixed over all the states (no rescaling when the slider moves)
    xs = np.concatenate([np.asarray(t.x, float) for t in traces if t.x is not None])
    ys = np.concatenate([np.asarray(t.y, float) for t in traces if t.y is not None])
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    pad = 0.03 * (ys.max() - ys.min())
    xr, yr = [float(xs.min()), float(xs.max())], [float(ys.min() - pad), float(ys.max() + pad)]
    titles =[[f"{nm}: {b[4]}" for nm in names] for b in built]
    buttons = [dict(label=nm, method="skip", args=[None]) for nm in names]
    sliders = []
    if len(built) > 1:
        sliders = [dict(active=0, currentvalue=dict(prefix="state: ", font=dict(size=13)), pad=dict(t=50),
                        steps=[dict(label=(f"{meta[p_] / 1e3:.1f} km{iter_tag(p_).replace(', Newton it ', ' it ')}" if meta[p_] is not None else os.path.basename(p_)),
                                    method="skip", args=[None], value=str(k)) for k, p_ in enumerate(paths)])]
    fig.update_layout(title=titles[0][0], template="plotly_white",
                      xaxis=dict(title="x (km)", range=xr, autorange=False),
                      yaxis=dict(title="z (m)", range=yr, autorange=False),
                      updatemenus=[dict(type="buttons", direction="right", buttons=buttons, x=0.0, y=1.06,
                                        xanchor="left", yanchor="bottom", showactive=True, active=0)],
                      sliders=sliders,
                      legend=dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom", bgcolor="rgba(255,255,255,0.7)"))
    import json
    # the buttons (field) and the slider (state) together: a small script keeps both choices and sets the visibility
    js = """
var gd = document.getElementById('{plot_id}');
var NF = %d, NFT = %d, UJ = %d, PER = %s, TITLES = %s, cur = {s: 0, f: 0};
var LOG = false, FIELD = true;       // FIELD: the colour field (temperature / u / w) shown; off: the surface lines only
function show() {
  var v = [], t = (cur.f === UJ && LOG) ? NF : cur.f;      // u with "log u" on: the log trace
  for (var s = 0; s < PER.length; s++)
    for (var j = 0; j < PER[s]; j++) v.push(s === cur.s && (j >= NFT || (FIELD && j === t)));
  Plotly.restyle(gd, {visible: v});
  Plotly.relayout(gd, {'title.text': TITLES[cur.s][cur.f] + ((cur.f === UJ && LOG) ? ' (log scale)' : '')});
}
gd.on('plotly_buttonclicked', function (e) { cur.f = e.active; show(); });
gd.on('plotly_sliderchange', function (e) { cur.s = parseInt(e.step.value); show(); });
var box = document.createElement('div');      // above the plot: the field toggle (+ the speed's limits)
box.style.cssText = 'font: 13px sans-serif; margin: 6px 0 0 80px;';
box.innerHTML = '<label><input id="ess_field" type="checkbox" checked> colour field</label> &nbsp; ';
gd.parentNode.insertBefore(box, gd);
document.getElementById('ess_field').addEventListener('change', function () {
  FIELD = document.getElementById('ess_field').checked; show(); });
if (UJ >= 0) {              // the speed's colour limits and the log toggle
  var lin = [], lg = [], o = 0;
  for (var s = 0; s < PER.length; s++) { lin.push(o + UJ); lg.push(o + NF); o += PER[s]; }
  box.insertAdjacentHTML('beforeend', 'u max (m/yr) <input id="ess_umax" type="number" step="any" style="width: 7em" value="%s"> ' +
    '&nbsp; u min, log scale (m/yr) <input id="ess_umin" type="number" step="any" style="width: 7em" value="%s"> ' +
    '&nbsp; <label><input id="ess_ulog" type="checkbox"> log u</label>');
  var apply = function () {
    var hi = parseFloat(document.getElementById('ess_umax').value), lo = parseFloat(document.getElementById('ess_umin').value);
    LOG = document.getElementById('ess_ulog').checked;
    if (hi > 0) {
      Plotly.restyle(gd, {zmax: hi}, lin);
      var l0 = Math.log10(lo > 0 && lo < hi ? lo : hi / 1e4), l1 = Math.log10(hi), tv = [], tt = [];
      for (var k = Math.floor(l0); k <= Math.ceil(l1); k++) { tv.push(k); tt.push(String(Math.pow(10, k))); }
      Plotly.restyle(gd, {zmin: l0, zmax: l1, 'colorbar.tickvals': [tv], 'colorbar.ticktext': [tt]}, lg);
    }
    if (cur.f === UJ) show();
  };
  ['ess_umax', 'ess_umin', 'ess_ulog'].forEach(function (id) { document.getElementById(id).addEventListener('change', apply); });
}
""" % (nf, nf + (1 if uj >= 0 else 0), uj, json.dumps(per), json.dumps(titles), f"{umax0:.4g}", f"{umin0:.3g}")
    out = out or (os.path.splitext(paths[0])[0] + ("_section.html" if len(paths) == 1 else "_sections.html"))
    fig.write_html(out, include_plotlyjs="cdn", post_script=js)
    print(f"-> {out} ({len(paths)} state(s), {os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
