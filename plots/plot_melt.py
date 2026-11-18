#!/usr/bin/env python3
"""The basal melt rate along x of flowline steady states (plotly), one line per state on one page: the fully temperate
steady state and the bed sweep's points (work/<case>/sweep_*/points/state_*.npz). The melt rate as the sweep computes
it (ess/spectrum.py melt_rate, ElmerStreams getmeltrate): m = -Temperature Loads / (boundary weight x rho_i x L), m/yr
of ice, positive melting, negative refreezing; on the temperate bed only (cold nodes: no line). The nodes across the
flowline (one element wide) are averaged per x. The grounding line of each state as a dotted line of its colour.

    python3 plots/plot_melt.py STATE.npz [STATE2.npz ...] [--x XMIN XMAX (km)] [--ymax M/YR] [--shelf]
                               [--params case/parameters_fine2x.lua] [--out FILE.html]

--shelf: the floating base too (default: the grounded bed only, the shelf's melt is the sea's temperature condition).
States without Temperature Loads (e.g. a Newton state saved without the heat diagnostics) are skipped with a message.
Default page: the first state's name + _melt.html.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ess.case import eval_lua_parameters  # noqa: E402
from ess.spectrum import _bed, melt_rate  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402


def opt(args, name, n=1, conv=float, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = [conv(a) for a in args[k + 1:k + 1 + n]]
    return args[:k] + args[k + 1 + n:], (v[0] if n == 1 else v)


def main():
    args = sys.argv[1:]
    args, xwin = opt(args, "--x", n=2)
    args, ymax = opt(args, "--ymax")
    args, params = opt(args, "--params", conv=str,
                       default=os.path.join(os.path.dirname(HERE), "case", "parameters_fine2x.lua"))
    args, out = opt(args, "--out", conv=str)
    shelf = "--shelf" in args
    args = [a for a in args if a != "--shelf"]
    if not args:
        raise SystemExit(__doc__)
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    P = eval_lua_parameters(params)
    rows = []
    for path in dict.fromkeys(args):
        s = load_state(path)
        d = float(s.meta["sweep_d"]) if "sweep_d" in s.meta else None
        g, xpos, grounded = _bed(s.frozen, s.diag)
        m = melt_rate(s.diag, g, P["rhoi"], P["Lw"])
        if m is None:
            print(f"{path}: no Temperature Loads / boundary weights in the state: skipped")
            continue
        ct = s.frozen[find_name(s.frozen, "ColdTempMask")]
        ctv = dict(zip(ct[0].tolist(), ct[1]))
        temp = np.array([ctv.get(k, -1.0) > 0 for k in g.tolist()])
        keep = temp & (grounded | shelf)
        xk = np.round(xpos / 1e3, 4)
        ux = np.unique(xk)
        mm = np.full(len(ux), np.nan)
        for i, xc in enumerate(ux):
            sel = (xk == xc) & keep
            if sel.any():
                mm[i] = np.nanmean(m[sel])
        xgl = float(xpos[grounded].max()) / 1e3 if grounded.any() else None
        lab = f"d = {d / 1e3:.1f} km" if d is not None else os.path.basename(path)
        gm = grounded & temp
        print(f"{lab}: grounded temperate bed {int(gm.sum())} nodes, melt {np.nanmin(m[gm]):.4g} .. {np.nanmax(m[gm]):.4g} "
              f"m/yr, total {np.nansum(mm[ux <= (xgl or np.inf)][:-1] * np.diff(ux[ux <= (xgl or np.inf)]) * 1e3):.4g} "
              f"m^2/yr (grounded, per unit width); grounding line {xgl} km")
        rows.append((d, lab, ux, mm, xgl, path))
    if not rows:
        raise SystemExit("no state with a melt rate")
    rows.sort(key=lambda r: (r[0] is not None, -(r[0] or 0)))      # the plain states first, then d from long to short
    cols = sample_colorscale("Plasma", [0.05 + 0.85 * k / max(len(rows) - 1, 1) for k in range(len(rows))])
    fig = go.Figure()
    for (d, lab, ux, mm, xgl, _), c in zip(rows, cols):
        sel = np.ones(len(ux), bool) if not xwin else (ux >= xwin[0]) & (ux <= xwin[1])
        fig.add_trace(go.Scatter(x=ux[sel], y=mm[sel], mode="lines", name=lab, legendgroup=lab, line=dict(color=c, width=1.6),
                                 connectgaps=False, hovertemplate=f"{lab}<br>x %{{x:.2f}} km<br>melt %{{y:.4g}} m/yr<extra></extra>"))
        if xgl is not None:
            fig.add_trace(go.Scatter(x=[xgl, xgl], y=[0, 1], yaxis="y2", mode="lines", legendgroup=lab, showlegend=False,
                                     line=dict(color=c, width=1, dash="dot"), hoverinfo="skip"))
    fig.add_hline(y=0, line=dict(color="grey", width=0.8))
    fig.update_layout(title="basal melt rate on the temperate bed (m/yr of ice; negative: refreezing); dotted: grounding lines",
                      template="plotly_white", xaxis=dict(title="x (km)", range=xwin),
                      yaxis=dict(title="melt rate (m/yr)", range=[-ymax, ymax] if ymax else None),
                      yaxis2=dict(overlaying="y", range=[0, 1], visible=False),
                      legend=dict(title="state (click to hide, double-click to isolate)"), hovermode="closest")
    out = out or os.path.splitext(args[0])[0] + "_melt.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
