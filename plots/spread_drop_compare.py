#!/usr/bin/env python3
"""The drop test's resolution series (configs/spread_drop_f*.toml): one page comparing the members.
  * the front radius against time (the azimuthal mean thickness H(r) falls below spread_thickness + --dh, default 10
    m), log-log, with Huppert's (1982) constant-volume similarity slope t^(1/8) for reference;
  * the azimuthal mean thickness H(r) of every member at a chosen time (a slider over the times all members have);
  * the front radius of each member against the finest one's, at the shared times.

    python3 plots/spread_drop_compare.py DIR [DIR ...] [--labels 2000,1000,...] [--dh 10] [--dr 100] [--out FILE.html]

DIR: a member's adapt/points folder (state_*.npz; a*_start.npz, the remeshed start, included).
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.statefile import find_name, load_state  # noqa: E402


def arg(args, name, default, conv=str):
    if name in args:
        k = args.index(name)
        v = conv(args[k + 1])
        del args[k:k + 2]
        return v
    return default


def profile(path, dr, H0):
    s = load_state(path)
    allc = {**s.diag, **s.frozen}
    X, Y = (dict(zip(*allc[find_name(allc, f"coordinate {k}")])) for k in (1, 2))
    blk = {b.name.lower(): b for b in s.layout.blocks}
    g = blk["zs"].gids
    zs = s.layout.split(s.x)[blk["zs"].name]
    B = dict(zip(*allc[find_name(allc, "bedrock")]))
    x, y = np.array([X[k] for k in g]), np.array([Y[k] for k in g])
    H = zs - np.array([B[k] for k in g])
    return float(s.meta.get("time", np.nan)), x, y, H


def main():
    args = sys.argv[1:]
    labels = arg(args, "--labels", None)
    dh = arg(args, "--dh", 10.0, float)
    dr = arg(args, "--dr", 100.0, float)
    H0 = arg(args, "--H0", 100.0, float)
    out = arg(args, "--out", None)
    if not args:
        raise SystemExit(__doc__)
    labels = labels.split(",") if labels else [os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(d))))
                                                 or d for d in args]
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import sample_colorscale
    mem = []
    for d, lab in zip(args, labels):
        files = sorted(glob.glob(os.path.join(d, "state_*.npz")) + glob.glob(os.path.join(d, "a*_start.npz")))
        rows = []
        for f in files:
            t, x, y, H = profile(f, dr, H0)
            # the centre: the thickness-weighted centroid of the drop (H - H0)
            w = np.clip(H - H0, 0, None)
            xc, yc = (np.sum(w * x) / w.sum(), np.sum(w * y) / w.sum()) if w.sum() > 0 else (x.mean(), y.mean())
            r = np.hypot(x - xc, y - yc)
            edges = np.arange(0, r.max() + dr, dr)
            k = np.digitize(r, edges) - 1
            n = np.bincount(k, minlength=len(edges))
            Hm = np.bincount(k, H, minlength=len(edges)) / np.maximum(n, 1)
            Hm[n == 0] = np.nan
            rb = edges + dr / 2
            ok = np.isfinite(Hm)
            below = np.where(ok & (Hm - H0 < dh))[0]
            rf = float(rb[below[0]]) if len(below) else np.nan
            rows.append(dict(t=t, rb=rb[ok], H=Hm[ok], rf=rf))
        rows.sort(key=lambda r_: r_["t"])
        # one row per time (the remeshed start duplicates the previous chunk's end)
        uniq = {}
        for r_ in rows:
            uniq[round(r_["t"], 6)] = r_
        rows = [uniq[k] for k in sorted(uniq)]
        print(f"{lab}: {len(rows)} states, t {rows[0]['t']:g} .. {rows[-1]['t']:g} yr, front "
              f"{rows[0]['rf'] / 1e3:.3f} .. {rows[-1]['rf'] / 1e3:.3f} km" if rows else f"{lab}: no states")
        if rows:
            mem.append(dict(lab=lab, rows=rows))
    if not mem:
        raise SystemExit("no states")
    cols = sample_colorscale("Viridis", [0.1 + 0.8 * i / max(len(mem) - 1, 1) for i in range(len(mem))])
    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]], vertical_spacing=0.12,
                        subplot_titles=(f"front radius (H < H0 + {dh:g} m) against time",
                                        "front radius minus the finest member's (m)",
                                        "azimuthal mean thickness H(r) (slider: time)"))
    for m, c in zip(mem, cols):
        t = np.array([r_["t"] for r_ in m["rows"]])
        rf = np.array([r_["rf"] for r_ in m["rows"]])
        ok = (t > 0) & np.isfinite(rf)
        fig.add_trace(go.Scatter(x=t[ok], y=rf[ok] / 1e3, mode="lines+markers", name=m["lab"], legendgroup=m["lab"],
                                 line=dict(color=c), marker=dict(size=4)), row=1, col=1)
    ref = mem[-1]
    tr = {round(r_["t"], 6): r_["rf"] for r_ in ref["rows"]}
    for m, c in zip(mem, cols):
        tt = [r_["t"] for r_ in m["rows"] if round(r_["t"], 6) in tr and r_["t"] > 0]
        dv = [r_["rf"] - tr[round(r_["t"], 6)] for r_ in m["rows"] if round(r_["t"], 6) in tr and r_["t"] > 0]
        fig.add_trace(go.Scatter(x=tt, y=dv, mode="lines+markers", name=m["lab"], legendgroup=m["lab"],
                                 showlegend=False, line=dict(color=c), marker=dict(size=4)), row=1, col=2)
    t_all = [r_["t"] for r_ in ref["rows"] if r_["t"] > 0]
    if t_all:
        t1, r1 = t_all[-1], ref["rows"][-1]["rf"] / 1e3
        tl = np.array([min(t_all), max(t_all)])
        fig.add_trace(go.Scatter(x=tl, y=r1 * (tl / t1) ** 0.125, mode="lines", name="t^1/8", line=dict(dash="dot",
                                 color="grey")), row=1, col=1)
    # profiles at the times every member has
    shared = sorted(set.intersection(*[set(round(r_["t"], 6) for r_ in m["rows"]) for m in mem]))
    i0 = len(fig.data)
    for k, ts in enumerate(shared):
        for m, c in zip(mem, cols):
            r_ = next(r for r in m["rows"] if round(r["t"], 6) == ts)
            fig.add_trace(go.Scatter(x=r_["rb"] / 1e3, y=r_["H"], mode="lines", name=m["lab"], legendgroup=m["lab"],
                                     showlegend=False, line=dict(color=c, width=1.5), visible=k == len(shared) - 1),
                          row=2, col=1)
    nm = len(mem)
    steps = [dict(label=f"{ts:,.0f}", method="restyle",
                  args=[{"visible": [j // nm == k for j in range(nm * len(shared))]},
                        list(range(i0, i0 + nm * len(shared)))]) for k, ts in enumerate(shared)]
    fig.update_xaxes(title_text="t (yr)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="front radius (km)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="t (yr)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="r (km)", row=2, col=1)
    fig.update_yaxes(title_text="H (m)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=950, title="drop test: the margin-resolution series",
                      sliders=[dict(active=max(len(shared) - 1, 0), currentvalue=dict(prefix="profiles at t (yr) = "),
                                    steps=steps, pad=dict(t=40))] if shared else [])
    out = out or os.path.join(os.getcwd(), "spread_drop_compare.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
