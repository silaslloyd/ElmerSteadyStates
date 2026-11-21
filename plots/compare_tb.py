#!/usr/bin/env python3
"""The basal temperature (T - 273.15 K, grounded bed nodes, along x) of flowline states against the SIA profile's
(plots/data/sia_profile.npz: Tb dimensionless, times Tscale = 5 K; x times the state's sia_scale_x, written by
jobs/sia_start.py, or --x-scale-km). Prints a table every --every km and writes a page (one line per state + the SIA).

    python3 plots/compare_tb.py STATE.npz [STATE2.npz ...] [--x-scale-km 518.2] [--every 50] [--out FILE.html]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ess.spectrum import _bed, basal_temperature  # noqa: E402
from ess.statefile import load_state  # noqa: E402

T0, TSCALE = 273.15, 5.0


def opt(args, name, conv=float, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    return args[:k] + args[k + 2:], conv(args[k + 1])


def main():
    args = sys.argv[1:]
    args, xs_km = opt(args, "--x-scale-km")
    args, every = opt(args, "--every", default=50.0)
    args, out = opt(args, "--out", conv=str)
    args, sia = opt(args, "--sia", conv=str, default=os.path.join(HERE, "data", "sia_profile.npz"))
    if not args:
        raise SystemExit(__doc__)
    z = np.load(sia)
    xd, bd, hd, tb = z["x"], z["bed"], z["h"], z["Tb"]
    fl = hd * 920.0 / 1003.27 < -bd
    i = int(np.argmax(fl)) if fl.any() else len(xd) - 1
    xd, tb = xd[:i + 1], tb[:i + 1] * TSCALE
    curves = []
    X = xs_km * 1e3 if xs_km else None
    for p in args:
        s = load_state(p)
        g, xpos, grounded = _bed(s.frozen, s.diag)
        T = basal_temperature(s.layout, s.x, g) - T0
        o = np.argsort(xpos[grounded])
        xg, Tg = xpos[grounded][o] / 1e3, T[grounded][o]
        xu, k = np.unique(np.round(xg, 3), return_index=True)
        curves.append((os.path.basename(p), xu, Tg[k]))
        if X is None:
            X = float(s.meta.get("sia_scale_x", np.nan))
    if not np.isfinite(X):
        raise SystemExit("no x scale: give --x-scale-km (or a state written by jobs/sia_start.py)")
    xk = xd * X / 1e3
    print(f"x scale {X / 1e3:.1f} km; T_b - 273.15 (K): SIA (Tb x {TSCALE:g} K) and the states")
    print(f"{'x km':>8} {'SIA':>8} " + " ".join(f"{n[:18]:>18}" for n, _, _ in curves))
    for xx in np.arange(0.0, xk[-1] + 1e-6, every):
        row = f"{xx:8.0f} {np.interp(xx, xk, tb):8.3f} "
        row += " ".join(f"{np.interp(xx, cx, cT):18.3f}" for _, cx, cT in curves)
        print(row)
    import plotly.graph_objects as go
    fig = go.Figure([go.Scatter(x=xk, y=tb, name=f"SIA (x scale {X / 1e3:.0f} km)", line=dict(color="#8e44ad", dash="dash"))]
                    + [go.Scatter(x=cx, y=cT, name=n) for n, cx, cT in curves])
    fig.update_layout(template="plotly_white", title="basal temperature, grounded bed (T - 273.15 K)",
                      xaxis_title="x (km)", yaxis_title="T_b - T_0 (K)")
    out = out or os.path.splitext(args[0])[0] + "_tb.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
