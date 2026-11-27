#!/usr/bin/env python3
"""
One plotly page per sweep with two plots: all the sweep points' surface profiles (Zs over x) and all their bed temperatures
(temperature of the lowest node of each column over x, K), one line per point, coloured by the patch length d.

  python3 plots/sweep_profiles.py STATE [STATE ...] --out page.html [--title TEXT]
STATE: npz states (the first, e.g. steady_temperate.npz, is drawn as d = whole grounded bed when it has no meta::sweep_d).
Needs numpy and plotly.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from plot_state import load, on_side  # noqa: E402
from ess.statefile import find_name  # noqa: E402


def one(path):
    """(d km or None, x_gl km or None, x of Zs, Zs, x of bed, bed temperature)"""
    layout, fields, diag, frozen, _, _ = load(path, 0)
    coords = {f"coordinate {c}": diag[find_name(diag, f"coordinate {c}")] for c in (1, 2, 3)}
    gx, X = coords["coordinate 1"]
    gz, Z = coords["coordinate 3"]
    xs = lambda g: X[np.searchsorted(gx, g)] / 1e3
    zs_ = lambda g: Z[np.searchsorted(gz, g)]
    nm = next(n for n in layout.names if n.lower() == "zs")
    g = next(b.gids for b in layout.blocks if b.name == nm)
    s = on_side(g, coords)
    x = xs(g[s])
    o = np.argsort(x)
    surf = (x[o], fields[nm][s][o])
    tn = next(n for n in layout.names if n.lower() == "temperature")
    g = next(b.gids for b in layout.blocks if b.name == tn)
    s = on_side(g, coords)
    x, z, T = np.round(xs(g[s]), 3), zs_(g[s]), fields[tn][s]
    ux = np.unique(x)
    bx, bt = [], []
    for u in ux:
        m = x == u
        bx.append(u)
        bt.append(T[m][np.argmin(z[m])])
    with np.load(path) as zz:
        d = float(zz["meta::sweep_d"]) / 1e3 if "meta::sweep_d" in zz.files else None
        xgl = float(zz["meta::x_gl"]) / 1e3 if "meta::x_gl" in zz.files else None
    return d, xgl, surf, (np.array(bx), np.array(bt))


def main():
    a = sys.argv[1:]
    out, title = "sweep_profiles.html", "sweep"
    if "--out" in a:
        i = a.index("--out"); out = a[i + 1]; del a[i:i + 2]
    if "--title" in a:
        i = a.index("--title"); title = a[i + 1]; del a[i:i + 2]
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    from plotly.subplots import make_subplots
    pts = []
    for p in a:
        try:
            pts.append((p,) + one(p))
        except Exception as e:
            print(f"skip {p}: {e}")
    if not pts:
        sys.exit("no states")
    xgl = next((q[2] for q in pts if q[2] is not None), None)
    known = [q[1] for q in pts if q[1] is not None]
    top = xgl if xgl is not None else (max(known) + 25 if known else 0.0)   # the whole grounded bed
    pts = [(p, d if d is not None else top, xg, s, b) for p, d, xg, s, b in pts]
    pts.sort(key=lambda q: -(q[1] if q[1] is not None else 0))
    ds = [q[1] for q in pts]
    lo, hi = min(ds), max(ds)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("surface elevation Zs (m)", "bed temperature (K)"))
    seen = set()
    for p, d, xg, s, b in pts:
        c = sample_colorscale("Viridis", [0.0 if hi == lo else (d - lo) / (hi - lo)])[0]
        lab = f"d = {d:.1f} km"
        if lab in seen:
            lab += " (" + os.path.basename(p) + ")"
        seen.add(lab)
        fig.add_trace(go.Scatter(x=s[0], y=s[1], mode="lines", name=lab, legendgroup=lab, line=dict(color=c, width=1.5)), 1, 1)
        fig.add_trace(go.Scatter(x=b[0], y=b[1], mode="lines", name=lab, legendgroup=lab, showlegend=False, line=dict(color=c, width=1.5)), 2, 1)
    fig.update_xaxes(title_text="x (km)", row=2, col=1)
    fig.update_layout(title=f"{title}: {len(pts)} sweep points, d {hi:.1f} to {lo:.1f} km", height=900,
                      legend=dict(title="patch length d"), hovermode="x unified")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"{out}: {len(pts)} points")


if __name__ == "__main__":
    main()
