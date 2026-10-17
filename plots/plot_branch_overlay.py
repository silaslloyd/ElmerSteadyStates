#!/usr/bin/env python3
"""
All states of a continuation branch on top of each other, coloured along the branch: basal
temperature and surface along x (or their change from the first state).

Colour follows the branch order (arclength), not Tsurf: past a fold Tsurf runs backwards, and
colouring by Tsurf would give the two sides of the fold the same colour. Hover shows each line's
point and Tsurf; the colour bar is labelled with Tsurf.

Press Run in an IDE (settings below), or:
    python plots/plot_branch_overlay.py work/land_margin/branch_flux work/land_margin/steady_p268.15_flux.npz
Needs numpy and plotly.
"""
import glob
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from plot_branch_states import extract, repo_path  # noqa: E402

# ---- settings for pressing Run (paths relative to the repository)
BRANCH = "work/land_margin/branch_flux"
START = None             # optional extra state drawn first (None: only the branch points)
MODE = "absolute"        # "absolute", or "difference": change from the first state
COLORSCALE = "Viridis"
SHOW = True


def main():
    bdir = repo_path(sys.argv[1] if len(sys.argv) > 1 else BRANCH)
    start = repo_path(sys.argv[2] if len(sys.argv) > 2 else START)
    files = ([start] if start and os.path.exists(start) else []) + \
        sorted(glob.glob(os.path.join(bdir, "points", "state_*.npz")))
    if not files:
        raise SystemExit(f"no states found in {bdir}/points")
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    from plotly.subplots import make_subplots

    S = [extract(f) for f in files]
    n = len(S)
    print(f"{n} states: Tsurf = " + ", ".join(f"{s['p']:.4f}" for s in S))
    frac = np.linspace(0, 1, n) if n > 1 else np.zeros(1)
    colours = sample_colorscale(COLORSCALE, list(frac))
    diff = MODE == "difference"
    ref = S[0]

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, shared_xaxes=True, subplot_titles=(
        "basal temperature" + (" change from the first state (K)" if diff else " (K)"),
        "surface elevation" + (" change from the first state (m)" if diff else " (m)")))
    for k, (s, c) in enumerate(zip(S, colours)):
        tb = s["Tb"] - np.interp(s["xb"], ref["xb"], ref["Tb"]) if diff else s["Tb"]
        lab = f"{k}: {s['name']}<br>Tsurf {s['p']:.4f} K"
        fig.add_trace(go.Scatter(x=s["xb"] / 1e3, y=tb, mode="lines", line=dict(color=c, width=1.5),
                                 name=f"{s['p']:.4f}", legendgroup=str(k), showlegend=False,
                                 hovertemplate=lab + "<br>x %{x:.1f} km<br>%{y:.4f}<extra></extra>"), 1, 1)
        if "xs" in s:
            zs = s["zs"] - np.interp(s["xs"], ref["xs"], ref["zs"]) if diff else s["zs"]
            fig.add_trace(go.Scatter(x=s["xs"] / 1e3, y=zs, mode="lines", line=dict(color=c, width=1.5),
                                     legendgroup=str(k), showlegend=False,
                                     hovertemplate=lab + "<br>x %{x:.1f} km<br>%{y:.2f}<extra></extra>"), 2, 1)
    if not diff:
        fig.add_hline(y=273.15, line=dict(dash="dot", color="grey"), row=1, col=1)

    # colour bar: an invisible marker trace carrying the scale, ticks labelled with Tsurf
    nt = min(n, 8)
    ti = np.unique(np.round(np.linspace(0, n - 1, nt)).astype(int))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=COLORSCALE, cmin=0, cmax=max(n - 1, 1), color=[0], showscale=True,
                    colorbar=dict(title="Tsurf (K)<br>along the branch", tickvals=list(ti),
                                  ticktext=[f"{S[i]['p']:.3f}" for i in ti], len=0.9))), 1, 1)
    fig.update_xaxes(title_text="x (km)", row=2, col=1)
    fig.update_layout(title=f"{os.path.basename(os.path.normpath(bdir))}: {n} states, "
                            f"Tsurf {min(s['p'] for s in S):.4f}-{max(s['p'] for s in S):.4f} K",
                      height=800, template="plotly_white", hovermode="closest")
    out = os.path.join(bdir, f"overlay_{MODE}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
