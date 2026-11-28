#!/usr/bin/env python3
"""
The extrapolation predictor (ess/predictor.py extrapolate_values) on a sweep's solved points, no model runs: the profiles
of the NEXT patch lengths beyond the last solved point, --step km apart (default 12.5) down to the grounding line (or
--next N of them), each from all the solved points. --retro instead: every solved point predicted from the points before it (to compare with what
the sweep found).

  python3 plots/extrapolation_check.py STATE [STATE ...] [--orders 2] [--next N] [--step 12.5] [--margin 2] [--retro] [--out page.html]

Rule (surface Zs and bed temperature at the ice base, each at FIXED x): downstream of the new head (the new patch) the
latest point's value, unchanged; upstream of it every x with at least two earlier values where it was still cold
(upstream of that point's head, 2 km margin) is extrapolated in d: linearly through the two nearest points, the
quadratic (--orders 2) blended in where a third point was cold too (no steps). Between the last extrapolated x and the
new head: bed temperature the straight line to the latest value at the head; surface the temperate surface plus an
offset falling linearly to 0 at the head (never below the latest surface).

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


def _common(prev):
    """the temperate (common) surface: the solved points' values at their own heads, and the latest point's surface
    downstream of its head; a function of x"""
    px, pv = [], []
    for d, head, (xz, zs), _ in prev:
        ux, inv = np.unique(np.round(xz, 4), return_inverse=True)
        mz = np.bincount(inv, zs) / np.bincount(inv)
        px.append(head)
        pv.append(float(np.interp(head, ux, mz)))
    d0, h0, (xz, zs), _ = prev[0]
    t = xz >= h0
    px += list(xz[t])
    pv += list(zs[t])
    o = np.argsort(px)
    px, pv = np.array(px)[o], np.array(pv)[o]
    return lambda x: np.interp(x, px, pv)


def predict(prev, x_z, x_t, d, head, orders, margin):
    """prev: the solved points, the latest (nearest in d) first. Returns {'Zs': {...}, 'Tb': {...}} with x and the
    prediction per order (o<order>) and 'previous' (the latest point)."""
    ds, hs = [q[0] for q in prev], [q[1] for q in prev]
    res = {"Zs": dict(x=x_z), "Tb": dict(x=x_t)}
    # surface: the ANOMALY above the temperate surface at fixed x (saturating in d), falling to 0 at the head
    c = _common(prev)(x_z)
    A = np.array([np.where(x_z < q[1], np.maximum(_on(x_z, q[2]) - c, 0.0), 0.0) for q in prev])
    res["Zs"]["previous"] = _on(x_z, prev[0][2])
    for od in orders:
        a, _ = extrapolate_values(x_z, A, ds, hs, d, head, order=od, join=True, margin=margin, saturate=True)
        a = np.where(x_z < head, np.maximum(a, 0.0), 0.0)
        m = x_z < head
        a[m] = np.minimum.accumulate(a[m])               # the anomaly only falls towards the head (as in every solved point)
        res["Zs"][f"o{od}"] = np.maximum(c + a, res["Zs"]["previous"])   # cold ice is higher: never below the latest surface
    # bed temperature: the cold bed follows one envelope curve (all the points lie on it upstream of their heads), the
    # temperate mask applies at the new head; beyond the last extrapolated x the envelope continues, saturating to the melting value
    V = np.array([_on(x_t, q[3]) for q in prev])
    Tm = V.max()
    res["Tb"]["previous"] = V[0]
    for od in orders:
        out, deg = extrapolate_values(x_t, V, ds, hs, d, head, order=od, join=False, margin=margin, saturate=True)
        full = (deg >= 1) & (x_t < head)
        if full.any():
            ia = int(np.argmax(np.where(full, x_t, -np.inf)))
            xa, Ta = x_t[ia], min(out[ia], Tm)
            near = full & (x_t >= xa - 50.0)
            slope = np.polyfit(x_t[near], out[near], 1)[0] if near.sum() > 2 else 0.0
            k = max(slope, 0.0) / max(Tm - Ta, 1e-9)
            rest = (x_t > xa) & (x_t < head)
            out[rest] = Tm - (Tm - Ta) * np.exp(-k * (x_t[rest] - xa))
        out = np.minimum(out, Tm)
        out[x_t >= head] = V[0][x_t >= head]               # the new patch: unchanged (melting)
        res["Tb"][f"o{od}"] = out
    return res


def cases(pts, orders=(1, 2), margin=2.0):
    """for each point k >= 1: predicted from the longer patches before it; dict(d, head, n, Zs, Tb) with 'actual' added"""
    res = []
    for k in range(1, len(pts)):
        d, head = pts[k][0], pts[k][1]
        prev = pts[:k][::-1]
        x_z = np.sort(pts[k][2][0])
        x_t = np.sort(pts[k][3][0])
        c = dict(d=d, head=head, n=len(prev), **predict(prev, x_z, x_t, d, head, orders, margin))
        c["Zs"]["actual"] = _on(x_z, pts[k][2])
        c["Tb"]["actual"] = _on(x_t, pts[k][3])
        res.append(c)
    return res


def forward(pts, ds_new, orders=(2,), margin=2.0):
    """the extrapolation to patch lengths beyond the solved points: each from ALL the points (not chained), on the last
    point's x grid. dicts like cases() without 'actual'"""
    prev = pts[::-1]                                   # the shortest patch (the latest) first
    xgl = pts[-1][0] + pts[-1][1]
    x_z, x_t = np.sort(pts[-1][2][0]), np.sort(pts[-1][3][0])
    res = []
    for d in ds_new:
        head = xgl - d
        c = dict(d=d, head=head, n=len(prev), **predict(prev, x_z, x_t, d, head, orders, margin))
        res.append(c)
        o = f"o{orders[-1]}"
        print(f"d = {d:.1f} km (head {head:.1f} km): max change from the last point: Zs "
              f"{np.max(np.abs(c['Zs'][o] - c['Zs']['previous'])):.3g} m, bed T {np.max(np.abs(c['Tb'][o] - c['Tb']['previous'])):.3g} K")
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
    pts = profiles(a)
    if len(pts) < 2:
        sys.exit("fewer than two sweep points")
    if retro:                                          # every point predicted from the longer patches before it
        cs = cases(pts, orders, margin)
        table(cs, orders)
    else:                                              # the next patch lengths beyond the last solved point
        n = nxt if nxt else int(np.ceil(pts[-1][0] / step - 1e-9)) - 1          # down to the grounding line (d > 0)
        cs = forward(pts, [pts[-1][0] - step * (k + 1) for k in range(n)], orders, margin)
    page(pts, cs, orders, out)
    np.savez(os.path.splitext(out)[0] + ".npz", **{f"d{c['d']:.1f}_{n}_{k}": c[n][k] for c in cs for n in ("Zs", "Tb")
                                                   for k in c[n]})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
