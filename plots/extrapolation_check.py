#!/usr/bin/env python3
"""
Check of the extrapolation predictor (ess/predictor.py extrapolate_values) on a finished sweep, no model runs: every point
is predicted from the points before it (the longer patches) and compared with what the sweep found.

  python3 plots/extrapolation_check.py STATE [STATE ...] [--orders 1 2] [--npts 4] [--out page.html]

Rule (surface Zs and bed temperature at the ice base, each at FIXED x): downstream of the new head (the new patch) the
latest point's value, unchanged; upstream of it the polynomial in d (degree = order, lower when there are fewer points)
through that x's values in the last npts points where x was still cold (upstream of that point's head); nodes the head
has just passed (no cold history) keep the latest value.

One plotly page, a slider over the predicted point: Zs and bed temperature (actual, the previous point as it is, the
extrapolations), and below each the error (prediction - actual). Axes fixed over all the points. A table of rms errors
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.3, 0.2, 0.3, 0.2],
                        subplot_titles=("surface elevation Zs (m)", "Zs: prediction - actual (m)",
                                        "bed temperature at the ice base (K)", "bed temperature: prediction - actual (K)"))
    keys = ["previous"] + [f"o{o}" for o in orders]
    lab = {"previous": "previous point", **{f"o{o}": f"extrapolation, order {o}" for o in orders}}
    col = {"actual": "black", "previous": "#999999", "o1": "#1f77b4", "o2": "#d62728", "o3": "#2ca02c"}
    per = []                                            # traces per case
    lim = {r: [np.inf, -np.inf] for r in (1, 2, 3, 4)}
    for i, c in enumerate(cs):
        n0 = len(fig.data)
        for name, rv, re_ in (("Zs", 1, 2), ("Tb", 3, 4)):
            d = c[name]
            fig.add_trace(go.Scatter(x=d["x"], y=d["actual"], mode="lines", name="actual", legendgroup="actual",
                                     showlegend=(name == "Zs"), visible=i == 0, line=dict(color=col["actual"], width=2)), rv, 1)
            for key in keys:
                dash = "dot" if key == "previous" else "dash"
                fig.add_trace(go.Scatter(x=d["x"], y=d[key], mode="lines", name=lab[key], legendgroup=key,
                                         showlegend=(name == "Zs"), visible=i == 0,
                                         line=dict(color=col.get(key, "purple"), width=1.5, dash=dash)), rv, 1)
                fig.add_trace(go.Scatter(x=d["x"], y=d[key] - d["actual"], mode="lines", name=lab[key], legendgroup=key,
                                         showlegend=False, visible=i == 0,
                                         line=dict(color=col.get(key, "purple"), width=1.5, dash=dash)), re_, 1)
            for r_, v in ((rv, np.r_[d["actual"], *(d[k] for k in keys)]), (re_, np.r_[*(d[k] - d["actual"] for k in keys)])):
                v = v[np.isfinite(v)]
                lim[r_] = [min(lim[r_][0], v.min()), max(lim[r_][1], v.max())]
        per.append((n0, len(fig.data)))
    steps = []
    for i, c in enumerate(cs):
        vis = [False] * len(fig.data)
        for j in range(*per[i]):
            vis[j] = True
        steps.append(dict(label=f"{c['d']:.1f} km", method="update", args=[{"visible": vis}]))
    for r_, (lo, hi) in lim.items():
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        fig.update_yaxes(range=[lo - pad, hi + pad], row=r_, col=1)
    fig.update_xaxes(title_text="x (km)", row=4, col=1)
    fig.update_layout(template="plotly_white", height=1250, legend=dict(orientation="h", x=0.0, y=1.06, xanchor="left", yanchor="bottom"),
                      title="extrapolation check: each point predicted from the longer patches before it (slider: the predicted point's d)",
                      sliders=[dict(active=0, currentvalue=dict(prefix="predicted point d = "), pad=dict(t=40), steps=steps)])
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
    orders = [1, 2]
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
