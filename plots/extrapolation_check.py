#!/usr/bin/env python3
"""
The extrapolation predictor (ess/predictor.py extrapolate_values) on a sweep's solved points, no model runs: the profiles
of the NEXT patch lengths beyond the last solved point, --step km apart (default 12.5) down to the grounding line (or
--next N of them), each from all the solved points. --retro instead: every solved point predicted from the points before it (to compare with what
the sweep found).

  python3 plots/extrapolation_check.py STATE [STATE ...] [--orders 2] [--npts order+1] [--next N] [--step 12.5] [--margin 2] [--retro] [--out page.html]

Rule (surface Zs and bed temperature at the ice base, each at FIXED x): downstream of the new head (the new patch) the
latest point's value, unchanged; upstream of it the polynomial in d (degree = order, lower when there are fewer points)
through that x's values in the last npts points where x was still cold (upstream of that point's head); nodes the head
has just passed (not enough cold history) are not extrapolated: the profile there is the straight line from the
extrapolated part to the latest value at the new head.

One plotly page, a slider over the predicted point: Zs and bed temperature (actual, the previous point as it is, the
extrapolations), Axes fixed over all the points. A table of rms errors
(upstream of the new head) is printed. Also written beside the page: <page>.npz, the predicted profiles.
Needs numpy and plotly.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from ess.predictor import extrapolate_values  # noqa: E402


def profiles(paths):
    """[(d km, head km, (x, Zs), (x, bed T))] of the sweep points among paths, longest patch first, one per d"""
    from sweep_profiles import one
    out = {}
    for p in paths:
        try:
            d, xgl, s, b, _ = one(p)
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        if d is None or xgl is None:
            continue
        out.setdefault(round(d * 10), (d, xgl - d, s, b))
    return [out[k] for k in sorted(out, reverse=True)]


def _on(x, prof):
    """the profile (x, v) pooled over equal x and interpolated to x"""
    px, pv = prof
    ux, inv = np.unique(np.round(px, 4), return_inverse=True)
    mv = np.bincount(inv, pv) / np.bincount(inv)
    return np.interp(x, ux, mv)


def cases(pts, orders=(1, 2), npts=4, margin=2.0):
    """for each point k >= 1: dict(d, head, x and values of Zs / Tb: actual, previous, prediction per order)"""
    res = []
    for k in range(1, len(pts)):
        d, head = pts[k][0], pts[k][1]
        prev = pts[:k][::-1]                           # the latest (the longest d below the target... the closest) first
        c = dict(d=d, head=head, n=len(prev))
        for name, j in (("Zs", 2), ("Tb", 3)):
            x, act = pts[k][j]
            o = np.argsort(x)
            x, act = x[o], act[o]
            V = np.array([_on(x, q[j]) for q in prev])
            c[name] = dict(x=x, actual=act, previous=V[0])
            for od in orders:
                c[name][f"o{od}"], _ = extrapolate_values(x, V, [q[0] for q in prev], [q[1] for q in prev], d, head,
                                                          order=od, npts=npts, join=True, margin=margin, relative=(name == "Zs"))
                if name == "Zs":
                    c[name][f"o{od}"] = np.maximum(c[name][f"o{od}"], V[0])
                if name == "Tb":
                    c[name][f"o{od}"] = np.minimum(c[name][f"o{od}"], V.max())
        res.append(c)
    return res


def forward(pts, ds_new, orders=(2,), npts=4, margin=2.0):
    """the extrapolation to patch lengths beyond the solved points: from ALL the points (each target on its own,
    not chained), on the last point's x grid. dicts like cases() but without 'actual'"""
    prev = pts[::-1]                                   # the shortest patch (the latest) first
    xgl = pts[-1][0] + pts[-1][1]
    res = []
    for d in ds_new:
        head = xgl - d
        c = dict(d=d, head=head, n=len(prev))
        for name, j in (("Zs", 2), ("Tb", 3)):
            x = np.sort(pts[-1][j][0])
            V = np.array([_on(x, q[j]) for q in prev])
            c[name] = dict(x=x, previous=V[0])
            for od in orders:
                c[name][f"o{od}"], deg = extrapolate_values(x, V, [q[0] for q in prev], [q[1] for q in prev], d, head,
                                                            order=od, npts=npts, join=True, margin=margin, relative=(name == "Zs"))
                if name == "Zs":                           # cold ice is higher: never below the latest surface
                    c[name][f"o{od}"] = np.maximum(c[name][f"o{od}"], V[0])
                if name == "Tb":                           # never above the melting value the solved points reach
                    c[name][f"o{od}"] = np.minimum(c[name][f"o{od}"], V.max())
        res.append(c)
        print(f"d = {d:.1f} km (head {head:.1f} km): max change from the last point: Zs "
              f"{np.max(np.abs(c['Zs'][f'o{orders[-1]}'] - c['Zs']['previous'])):.3g} m, bed T "
              f"{np.max(np.abs(c['Tb'][f'o{orders[-1]}'] - c['Tb']['previous'])):.3g} K")
    return res


def table(cs, orders):
    names = ["previous"] + [f"order {o}" for o in orders]
    print(f"{'d km':>8} {'pts':>3}  " + "  ".join(f"Zs {n:>9}" for n in names) + "  |  " + "  ".join(f"Tb {n:>9}" for n in names))
    for c in cs:
        row = []
        for name in ("Zs", "Tb"):
            m = c[name]["x"] < c["head"]
            for key in ["previous"] + [f"o{o}" for o in orders]:
                e = c[name][key][m] - c[name]["actual"][m]
                row.append(f"{np.sqrt(np.mean(e ** 2)) if m.any() else np.nan:12.4g}")
        print(f"{c['d']:8.1f} {c['n']:3d}  " + "  ".join(row[:len(names)]) + "  |  " + "  ".join(row[len(names):]))


def page(pts, cs, orders, out):
    """one frame: every solved point's profile solid black, every extrapolated one dashed (one colour per order)"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("surface elevation Zs (m)", "bed temperature at the ice base (K)"))
    col = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    for k, q in enumerate(pts):
        for j, r_ in ((2, 1), (3, 2)):
            x, v = q[j]
            o = np.argsort(x)
            fig.add_trace(go.Scatter(x=x[o], y=v[o], mode="lines", name="solved", legendgroup="solved",
                                     showlegend=k == 0 and r_ == 1, line=dict(color="black", width=1.5),
                                     hovertemplate=f"solved d = {q[0]:.1f} km<br>x = %{{x:.1f}} km<br>%{{y:.4g}}<extra></extra>"), r_, 1)
    first = True
    for c in cs:
        for name, r_ in (("Zs", 1), ("Tb", 2)):
            d = c[name]
            for od in orders:
                fig.add_trace(go.Scatter(x=d["x"], y=d[f"o{od}"], mode="lines", name=f"extrapolated, order {od}",
                                         legendgroup=f"o{od}", showlegend=first and r_ == 1,
                                         line=dict(color=col.get(od, "purple"), width=1.5, dash="dash"),
                                         hovertemplate=f"extrapolated d = {c['d']:.1f} km<br>x = %{{x:.1f}} km<br>%{{y:.4g}}<extra></extra>"), r_, 1)
        first = False
    fig.update_xaxes(title_text="x (km)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=900, margin=dict(t=60, r=200),
                      legend=dict(x=1.01, y=1.0, xanchor="left", yanchor="top"))
    fig.write_html(out, include_plotlyjs="cdn")


def main():
    a = sys.argv[1:]

    def opt(name, default, conv):
        nonlocal a
        if name in a:
            i = a.index(name)
            v = a[i + 1]
            del a[i:i + 2]
            return conv(v)
        return default
    out = opt("--out", "extrapolation_check.html", str)
    npts = opt("--npts", None, int)
    orders = [2]
    if "--orders" in a:
        i = a.index("--orders")
        j = i + 1
        orders = []
        while j < len(a) and not a[j].startswith("--") and a[j].isdigit():
            orders.append(int(a[j]))
            j += 1
        del a[i:j]
    margin = opt("--margin", 2.0, float)
    nxt = opt("--next", None, int)
    step = opt("--step", 12.5, float)
    retro = "--retro" in a
    a = [t for t in a if t != "--retro"]
    npts = npts or max(orders) + 1                     # exactly order + 1 states: one fit for the whole extrapolated zone
    pts = profiles(a)
    if len(pts) < 2:
        sys.exit("fewer than two sweep points")
    if retro:                                          # every point predicted from the longer patches before it
        cs = cases(pts, orders, npts, margin)
        table(cs, orders)
    else:                                              # the next patch lengths beyond the last solved point
        n = nxt if nxt else int(np.ceil(pts[-1][0] / step - 1e-9)) - 1          # down to the grounding line (d > 0)
        cs = forward(pts, [pts[-1][0] - step * (k + 1) for k in range(n)], orders, npts, margin)
    page(pts, cs, orders, out)
    np.savez(os.path.splitext(out)[0] + ".npz", **{f"d{c['d']:.1f}_{n}_{k}": c[n][k] for c in cs for n in ("Zs", "Tb")
                                                   for k in c[n]})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
