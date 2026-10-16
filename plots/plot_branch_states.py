#!/usr/bin/env python3
"""
Step through the states of a continuation branch with a slider: basal temperature and surface along
x, for the starting state and every branch point.

Press Run in an IDE (settings below), or:
    python plots/plot_branch_states.py work/land_flowline/branch_cold work/land_flowline/steady_p268.15_cold.npz
Needs numpy and plotly. Linear axes, fixed axis and colour ranges across the slider.
"""
import glob
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from ess.statefile import find_name, load_state  # noqa: E402

# ---- settings for pressing Run (paths relative to the repository)
BRANCH = "work/land_flowline/branch_cold"              # branch directory
START = "work/land_flowline/steady_p268.15_cold.npz"   # starting state (None: only the branch points)
Y_SIDE = "min"        # the flowline mesh is one element wide: nodes at y = min (or "max")
SHOW = True


def repo_path(p):
    return p if p is None or os.path.isabs(p) else os.path.join(REPO, p)


def extract(path):
    """Basal T(x), Zs(x) and the temperature section of one state file."""
    s = load_state(path)
    diag = s.diag
    coords = {c: diag[find_name(diag, f"coordinate {c}")] for c in (1, 2, 3)}
    gx, X = coords[1]
    gy, Y = coords[2]
    gz, Z = coords[3]
    side = Y.min() if Y_SIDE == "min" else Y.max()
    split = s.layout.split(s.x)
    tname = next(n for n in s.layout.names if n.lower() == "temperature")
    zname = next((n for n in s.layout.names if n.lower() == "zs"), None)
    gids = {b.name: b.gids for b in s.layout.blocks}

    def at(g, gc, vc):
        return vc[np.searchsorted(gc, g)]

    def on_side(g):
        return np.isclose(at(g, gy, Y), side)

    out = {"p": s.p, "name": os.path.basename(path)}
    # bed nodes = nodes carrying the frozen masks
    src = s.frozen or diag
    bkey = next(k for k in src if k.lower() in ("coldtempmask", "zb"))
    gb = src[bkey][0]
    gb = gb[on_side(gb)]
    xb = at(gb, gx, X)
    o = np.argsort(xb)
    out["xb"], out["Tb"] = xb[o], split[tname][np.searchsorted(gids[tname], gb)][o]
    if zname:
        gs = gids[zname][on_side(gids[zname])]
        xs = at(gs, gx, X)
        o = np.argsort(xs)
        out["xs"], out["zs"] = xs[o], split[zname][np.searchsorted(gids[zname], gs)][o]
    if "Tb_cold_max" in s.meta:
        out["Tb_cold_max"] = float(s.meta["Tb_cold_max"])
    return out


def main():
    bdir = repo_path(sys.argv[1] if len(sys.argv) > 1 else BRANCH)
    start = repo_path(sys.argv[2] if len(sys.argv) > 2 else START)
    files = ([start] if start and os.path.exists(start) else []) + \
        sorted(glob.glob(os.path.join(bdir, "points", "state_*.npz")))
    if not files:
        raise SystemExit(f"no states found (start {start}, branch {bdir}/points/state_*.npz)")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    S = [extract(f) for f in files]
    print(f"{len(S)} states: Tsurf = " + ", ".join(f"{s['p']:.3f}" for s in S))
    tb_rng = [min(s["Tb"].min() for s in S) - 0.2, max(273.3, max(s["Tb"].max() for s in S) + 0.1)]
    xmax = max(s["xb"].max() for s in S) / 1e3

    def traces(s):
        tr = [go.Scatter(x=s["xb"] / 1e3, y=np.round(s["Tb"], 4), mode="lines", line=dict(color="#c0392b"),
                         name="T bed", hovertemplate="x %{x:.1f} km<br>T %{y:.3f} K<extra></extra>")]
        tr.append(go.Scatter(x=s["xs"] / 1e3, y=np.round(s["zs"], 2), mode="lines", line=dict(color="#2c3e50"),
                             name="Zs", hovertemplate="x %{x:.1f} km<br>Zs %{y:.1f} m<extra></extra>")
                  if "xs" in s else go.Scatter(x=[], y=[]))
        return tr

    def title(s):
        extra = f", warmest cold-bed node {s['Tb_cold_max']:.3f} K" if "Tb_cold_max" in s else ""
        return f"{s['name']}: Tsurf = {s['p']:.4f} K{extra}"

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1,
                        subplot_titles=("basal temperature (K)", "surface elevation (m)"))
    for i, t in enumerate(traces(S[0])):
        fig.add_trace(t, i + 1, 1)
    fig.add_hline(y=273.15, line=dict(dash="dot", color="grey"), row=1, col=1)
    frames, steps = [], []
    for k, s in enumerate(S):
        frames.append(go.Frame(name=str(k), data=traces(s), traces=[0, 1], layout=dict(title_text=title(s))))
        steps.append(dict(method="animate", label=f"{s['p']:.3f}",
                          args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0))]))
    fig.frames = frames
    fig.update_xaxes(range=[0, xmax], row=1, col=1)
    fig.update_xaxes(range=[0, xmax], title_text="x (km)", row=2, col=1)
    fig.update_yaxes(range=tb_rng, row=1, col=1)
    zs_all = np.concatenate([s["zs"] for s in S if "zs" in s])
    fig.update_yaxes(range=[zs_all.min() - 50, zs_all.max() + 50], row=2, col=1)
    fig.update_layout(
        title=title(S[0]), height=800, template="plotly_white", showlegend=False, margin=dict(t=90, b=130),
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="Tsurf = ", xanchor="right"),
                      x=0.12, len=0.88, y=0, yanchor="top", pad=dict(t=50))],
        updatemenus=[dict(type="buttons", direction="left", x=0, y=0, xanchor="left", yanchor="top",
                          pad=dict(t=60), buttons=[
            dict(label="play", method="animate",
                 args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True,
                                  transition=dict(duration=0))]),
            dict(label="pause", method="animate",
                 args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])])
    out = os.path.join(bdir, "states.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
