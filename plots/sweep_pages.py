#!/usr/bin/env python3
"""One plotly page per sweep: every converged point's surface Zs(x) on one plot, every point's grounded basal
temperature T_b(x) - 273.15 K on a second plot below it (shared x), coloured by the patch length d.

    python3 plots/sweep_pages.py PATH [PATH ...] [--out DIR] [--recent-days N]

PATH: a sweep folder (sweep.csv + points/state_XXXXX.npz, as ess sweep / run_sweep_marine writes), a refined sweep's
folder (seg_*/sweep.csv: the segments merged into one sweep), or any folder above them (searched for sweep.csv; a seg_*
folder counts towards its parent). --recent-days N: only sweeps whose sweep.csv changed in the last N days.
Writes DIR/<sweep name>.html per sweep and DIR/index.html linking them (default DIR: sweep_pages next to the first PATH).

Legend entries are per point and toggle both plots together; hover shows d, the patch head and the grounding line.
Markers on the surfaces mark each point's patch head (x_gl - d).
"""
import argparse
import csv
import glob
import html
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ess.spectrum import _bed, basal_temperature, gl_position  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402

T0 = 273.15
# viridis, 9 stops (matplotlib's), for the per-point colours (no plotly.colors dependency)
_VIR = np.array([[68, 1, 84], [71, 45, 123], [59, 82, 139], [44, 114, 142], [33, 145, 140], [40, 174, 128],
                 [94, 201, 98], [173, 220, 48], [253, 231, 37]], float)


def viridis(u):
    u = float(np.clip(u, 0.0, 1.0)) * (len(_VIR) - 1)
    i = min(int(u), len(_VIR) - 2)
    c = _VIR[i] + (u - i) * (_VIR[i + 1] - _VIR[i])
    return "rgb({:.0f},{:.0f},{:.0f})".format(*c)


def column_profile(s, name):
    """a nodal block's value per column (x in m, sorted; the columns' nodes averaged); as jobs/sia_start.py"""
    gx, X = s.diag[find_name(s.diag, "coordinate 1")]
    blk = next(b for b in s.layout.blocks if b.name.lower() == name)
    v = s.x[s.layout.slices()[blk.name]]
    x = X[np.searchsorted(gx, blk.gids)]
    u, inv = np.unique(np.round(x, 3), return_inverse=True)
    return u, np.bincount(inv, v) / np.bincount(inv)


def converged(folder):
    """the converged (d km, state path) of one sweep folder (as jobs/sweep_refined.py converged)"""
    f = os.path.join(folder, "sweep.csv")
    if not os.path.exists(f):
        return []
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    return [(float(r["d_km"]), os.path.join(folder, "points", f"state_{int(round(float(r['d_km']) * 10)):05d}.npz"))
            for r in rows if str(r.get("converged")) == "1"]


def find_sweeps(paths, recent_days=None):
    """{sweep folder: [its sweep.csv folders]}: a seg_* folder counts towards its parent"""
    found = {}
    now = time.time()
    for p in paths:
        p = os.path.normpath(p)
        csvs = [os.path.join(dp, "sweep.csv") for dp, _, fs in os.walk(p) if "sweep.csv" in fs]
        for c in sorted(csvs):
            if recent_days is not None and now - os.path.getmtime(c) > recent_days * 86400:
                continue
            d = os.path.dirname(c)
            root = os.path.dirname(d) if os.path.basename(d).startswith("seg_") else d
            found.setdefault(root, []).append(d)
    return found


def load_points(folders):
    pts = {}
    for f in folders:
        for d, p in converged(f):
            if os.path.exists(p):
                pts[round(d, 1)] = p                # a segment's start point appears in two segments: once
    out = []
    for d in sorted(pts, reverse=True):
        try:
            s = load_state(pts[d])
            xgl = gl_position(s.frozen, s.diag) / 1e3
            xs, zs = column_profile(s, "zs")
            g, xpos, gr = _bed(s.frozen, s.diag)
            Tb = basal_temperature(s.layout, s.x, g) - T0
            o = np.argsort(xpos[gr])
            xu, ku = np.unique(np.round(xpos[gr][o] / 1e3, 3), return_index=True)
            zb = column_profile(s, "zb")
        except Exception as e:                       # a damaged / foreign file: skipped, not fatal
            print(f"  {pts[d]}: skipped ({type(e).__name__}: {e})")
            continue
        out.append(dict(d=d, xgl=xgl, head=xgl - d, xs=xs / 1e3, zs=zs, xt=xu, tb=Tb[gr][o][ku],
                        xb=zb[0] / 1e3, zb=zb[1]))
    return out


def page(name, pts, path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    ds = np.array([p["d"] for p in pts])
    dmin, dmax = float(ds.min()), float(ds.max())
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("surface Zs", "basal temperature (grounded bed)"))
    for p in pts:
        c = viridis((p["d"] - dmin) / (dmax - dmin) if dmax > dmin else 1.0)
        lab = f"d {p['d']:g} km"
        info = f"d {p['d']:g} km, head {p['head']:.1f} km, GL {p['xgl']:.1f} km"
        fig.add_trace(go.Scatter(x=p["xs"], y=p["zs"], mode="lines", name=lab, legendgroup=lab,
                                 line=dict(color=c, width=1.5),
                                 hovertemplate=f"{info}<br>x %{{x:.1f}} km<br>Zs %{{y:.1f}} m<extra></extra>"),
                      row=1, col=1)
        zh = float(np.interp(p["head"], p["xs"], p["zs"]))
        fig.add_trace(go.Scatter(x=[p["head"]], y=[zh], mode="markers", name=lab, legendgroup=lab, showlegend=False,
                                 marker=dict(color=c, size=7, symbol="diamond"),
                                 hovertemplate=f"head of the patch<br>{info}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=p["xt"], y=p["tb"], mode="lines", name=lab, legendgroup=lab, showlegend=False,
                                 line=dict(color=c, width=1.5),
                                 hovertemplate=f"{info}<br>x %{{x:.1f}} km<br>T_b %{{y:.3f}} °C<extra></extra>"),
                      row=2, col=1)
    nb = pts[-1]
    fig.add_trace(go.Scatter(x=nb["xb"], y=nb["zb"], mode="lines", name=f"Zb (d {nb['d']:g} km)",
                             line=dict(color="#888", width=1, dash="dot"), visible="legendonly"), row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color="red", width=1, dash="dash"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale="Viridis", cmin=dmin, cmax=dmax, color=[dmin], showscale=True,
                                         colorbar=dict(title="d (km)", len=0.45, y=0.78, x=1.12))), row=1, col=1)
    fig.update_yaxes(title_text="Zs (m)", row=1, col=1)
    fig.update_yaxes(title_text="T_b − 273.15 (K)", row=2, col=1)
    fig.update_xaxes(title_text="x (km)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=900, hovermode="closest",
                      title=f"{name}: {len(pts)} converged points, d {dmax:g} → {dmin:g} km",
                      legend=dict(groupclick="togglegroup"))
    fig.write_html(path, include_plotlyjs="cdn")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--recent-days", type=float)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.normpath(a.paths[0])) or ".", "sweep_pages")
    os.makedirs(out, exist_ok=True)
    sweeps = find_sweeps(a.paths, a.recent_days)
    if not sweeps:
        raise SystemExit("no sweep.csv found" + (f" changed in the last {a.recent_days:g} days" if a.recent_days else ""))
    base = os.path.commonpath([os.path.abspath(r) for r in sweeps]) if len(sweeps) > 1 else None
    made = []
    for root, folders in sorted(sweeps.items()):
        rel = os.path.relpath(os.path.abspath(root), base) if base else os.path.basename(os.path.abspath(root))
        name = rel.replace(os.sep, "__")
        pts = load_points(folders)
        if not pts:
            print(f"{root}: no converged points yet, no page")
            continue
        fn = os.path.join(out, f"{name}.html")
        page(name, pts, fn)
        made.append((name, len(pts), pts[0]["d"], pts[-1]["d"]))
        print(f"{root}: {len(pts)} points (d {pts[0]['d']:g} -> {pts[-1]['d']:g} km) -> {fn}")
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>sweeps</title>"
                "<body style='font-family:sans-serif'><h2>sweeps</h2><ul>")
        for name, n, d0, d1 in made:
            f.write(f"<li><a href='{html.escape(name)}.html'>{html.escape(name)}</a> — {n} points, d {d0:g} → {d1:g} km</li>")
        f.write(f"</ul><p style='color:#888'>made {time.strftime('%Y-%m-%d %H:%M')}</p></body>")
    print(f"-> {os.path.join(out, 'index.html')} ({len(made)} pages)")


if __name__ == "__main__":
    main()
