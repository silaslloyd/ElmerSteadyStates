#!/usr/bin/env python3
"""
One plotly page with three plots: all a sweep's points' surface profiles (Zs over x), all their bed temperatures (the ice base,
K) and all their steady-state hydrology (the water sheet thickness on the bed, m), one line per point, coloured by the patch length d. A slider picks the sweep by its parameter value (surface temperature Tsurf).

  python3 plots/sweep_profiles.py STATE [STATE ...] [-- STATE [STATE ...]] ... --out page.html
One sweep per group of states, groups separated by --; the first state of a group (e.g. steady_temperate.npz) is drawn as d = whole
grounded bed when it has no meta::sweep_d. Tsurf is the state's p. Groups without a readable state are left out.
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
    # the ice base Zb (layout block, else a frozen / diagnostic record): the bed temperature is the ice's lowest node, not the rock's
    zb = None
    nb = next((n for n in layout.names if n.lower() == "zb"), None)
    if nb is not None:
        gb, vb = next(b.gids for b in layout.blocks if b.name == nb), fields[nb]
    else:
        src = {**diag, **frozen}
        kb = next((k for k in src if k.lower() == "zb"), None)
        gb, vb = src[kb] if kb is not None else (None, None)
    if gb is not None:
        sb = on_side(gb, coords)
        o = np.argsort(xs(gb[sb]))
        zb = (xs(gb[sb])[o], vb[sb][o])
        ice = z >= np.interp(x, *zb) - 0.5
        x, z, T = x[ice], z[ice], T[ice]
    else:
        print(f"{path}: no Zb, the lowest temperature node of each column is used")
    ux = np.unique(x)
    bx, bt = [], []
    for u in ux:
        m = x == u
        bx.append(u)
        bt.append(T[m][np.argmin(z[m])])
    hyd = None                                   # the water sheet (SheetSolverhw): a frozen field on the bed nodes
    hk = next((k for k in frozen if k.lower() == "water sheet thickness"), None)
    if hk is not None:
        gh, vh = frozen[hk]
        sh = on_side(gh, coords)
        xh = xs(gh[sh])
        o = np.argsort(xh)
        hyd = (xh[o], np.asarray(vh, float)[sh][o])
    with np.load(path) as zz:
        d = float(zz["meta::sweep_d"]) / 1e3 if "meta::sweep_d" in zz.files else None
        xgl = float(zz["meta::x_gl"]) / 1e3 if "meta::x_gl" in zz.files else None
    return d, xgl, surf, (np.array(bx), np.array(bt)), hyd


def load_group(paths):
    pts = []
    for p in paths:
        try:
            pts.append((p,) + one(p))
        except Exception as e:
            print(f"skip {p}: {e}")
    if not pts:
        return None
    with np.load(next(q[0] for q in pts)) as zz:
        tsurf = float(zz["p"])
    xgl = next((q[2] for q in pts if q[2] is not None), None)
    known = [q[1] for q in pts if q[1] is not None]
    top = xgl if xgl is not None else (max(known) + 25 if known else 0.0)   # the whole grounded bed
    pts = [(p, d if d is not None else top, s, b, h) for p, d, xg, s, b, h in pts]
    pts.sort(key=lambda q: -q[1])
    nh = sum(q[4] is None for q in pts)
    if nh:
        print(f"Tsurf {tsurf:.2f}: {nh} of {len(pts)} states have no water sheet thickness record")
    return tsurf, pts


def main():
    a = sys.argv[1:]
    out = "sweep_profiles.html"
    if "--out" in a:
        i = a.index("--out"); out = a[i + 1]; del a[i:i + 2]
    groups, cur = [], []
    for t in a + ["--"]:
        if t == "--":
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append(t)
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    from plotly.subplots import make_subplots
    sweeps = [g for g in (load_group(g) for g in groups) if g]
    if not sweeps:
        sys.exit("no states")
    sweeps.sort(key=lambda t: t[0])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("surface elevation Zs (m)", "bed temperature at the ice base (K)", "water sheet thickness on the bed, steady-state hydrology (m)"))
    owner = []                                   # per trace: the sweep it belongs to
    for k, (ts, pts) in enumerate(sweeps):
        ds = [q[1] for q in pts]
        lo, hi = min(ds), max(ds)
        seen = set()
        for p, d, s, b, h in pts:
            c = sample_colorscale("Viridis", [0.0 if hi == lo else (d - lo) / (hi - lo)])[0]
            lab = f"d = {d:.1f} km"
            if lab in seen:
                lab += " (" + os.path.basename(p) + ")"
            seen.add(lab)
            vis = k == 0
            fig.add_trace(go.Scatter(x=s[0], y=s[1], mode="lines", name=lab, legendgroup=lab, visible=vis, line=dict(color=c, width=1.5)), 1, 1)
            fig.add_trace(go.Scatter(x=b[0], y=b[1], mode="lines", name=lab, legendgroup=lab, showlegend=False, visible=vis, line=dict(color=c, width=1.5)), 2, 1)
            fig.add_trace(go.Scatter(x=h[0] if h is not None else [], y=h[1] if h is not None else [], mode="lines", name=lab, legendgroup=lab,
                                     showlegend=False, visible=vis, line=dict(color=c, width=1.5)), 3, 1)
            owner += [k, k, k]
    titles = [f"Tsurf = {ts:.2f} K: {len(pts)} sweep points, d {max(q[1] for q in pts):.1f} to {min(q[1] for q in pts):.1f} km" for ts, pts in sweeps]
    steps = [dict(method="update", label=f"{ts:.2f}", args=[{"visible": [o == k for o in owner]}, {"title": titles[k]}])
             for k, (ts, _) in enumerate(sweeps)]
    fig.update_xaxes(title_text="x (km)", row=3, col=1)
    # y axes fixed to the extremes over all sweeps (a little padding), so the slider does not rescale them
    for row, k in ((1, 2), (2, 3), (3, 4)):
        v = np.concatenate([np.asarray(q[k][1], float) for _, pts in sweeps for q in pts if q[k] is not None] or [np.zeros(1)])
        v = v[np.isfinite(v)]
        pad = 0.03 * (v.max() - v.min() or 1.0)
        fig.update_yaxes(range=[float(v.min() - pad), float(v.max() + pad)], row=row, col=1)
    fig.update_layout(title=titles[0], height=1250, legend=dict(title="patch length d"), hovermode="x unified",
                      sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="Tsurf (K) = "), pad=dict(t=40))])
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"{out}: {len(sweeps)} sweeps, Tsurf " + ", ".join(f"{ts:.2f}" for ts, _ in sweeps))


if __name__ == "__main__":
    main()
