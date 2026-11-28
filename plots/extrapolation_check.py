#!/usr/bin/env python3
"""
Check of the extrapolation predictor (ess/predictor.py extrapolate_values) on a finished sweep, no model runs: every point
is predicted from the points before it (the longer patches) and compared with what the sweep found.

  python3 plots/extrapolation_check.py STATE [STATE ...] [--orders 2] [--npts 4] [--out page.html]

Rule (surface Zs and bed temperature at the ice base, each at FIXED x): downstream of the new head (the new patch) the
latest point's value, unchanged; upstream of it the polynomial in d (degree = order, lower when there are fewer points)
through that x's values in the last npts points where x was still cold (upstream of that point's head); nodes the head
has just passed (no cold history) keep the latest value.

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


def cases(pts, orders=(1, 2), npts=4):
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
                                                          order=od, npts=npts)
        res.append(c)
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


def page(cs, orders, out):
    """one frame: every point's actual profile solid black, every extrapolated one dashed (one colour per order)"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("surface elevation Zs (m)", "bed temperature at the ice base (K)"))
    col = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
    first = True
    for c in cs:
        for name, r_ in (("Zs", 1), ("Tb", 2)):
            d = c[name]
            fig.add_trace(go.Scatter(x=d["x"], y=d["actual"], mode="lines", name="actual", legendgroup="actual",
                                     showlegend=first and r_ == 1, line=dict(color="black", width=1.5),
                                     hovertemplate=f"actual d = {c['d']:.1f} km<br>x = %{{x:.1f}} km<br>%{{y:.4g}}<extra></extra>"), r_, 1)
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
    npts = opt("--npts", 4, int)
    orders = [2]
    if "--orders" in a:
        i = a.index("--orders")
        j = i + 1
        orders = []
        while j < len(a) and not a[j].startswith("--") and a[j].isdigit():
            orders.append(int(a[j]))
            j += 1
        del a[i:j]
    pts = profiles(a)
    if len(pts) < 2:
        sys.exit("fewer than two sweep points")
    cs = cases(pts, orders, npts)
    table(cs, orders)
    page(cs, orders, out)
    np.savez(os.path.splitext(out)[0] + ".npz", **{f"d{c['d']:.1f}_{n}_{k}": c[n][k] for c in cs for n in ("Zs", "Tb")
                                                   for k in ["x", "actual", "previous"] + [f"o{o}" for o in orders]})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
