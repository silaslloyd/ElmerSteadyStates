#!/usr/bin/env python3
"""
The states of a continuation branch: basal temperature, surface and (when `ess velocity` has been
run on them) the along-flow and vertical velocities along x, for the starting state and every branch point, either one at a time with a slider (VIEW = "slider") or all on top of each
other coloured along the branch (VIEW = "overlay").

In the overlay the colour follows the branch order (arclength), not Tsurf: past a fold Tsurf runs
backwards, and colouring by Tsurf would give both sides of the fold the same colour. The colour bar
is labelled with Tsurf, and hovering over a line shows its point and Tsurf.

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
from plot_state import column_grid  # noqa: E402

# ---- settings for pressing Run (paths relative to the repository)
BRANCH = "work/land_flowline/branch_cold"              # branch directory
START = "work/land_flowline/steady_p268.15_cold.npz"   # starting state (None: only the branch points)
Y_SIDE = "min"        # the flowline mesh is one element wide: nodes at y = min (or "max")
SHOW = True

VIEW = "slider"       # "slider": one state at a time; "overlay": all states, coloured along the branch
MODE = "absolute"     # overlay only: "absolute", or "difference" (change from the first state)
COLORSCALE = "Viridis"
VEL_PLOT = "contour"  # slider only: "contour" (filled u, w sections in x-z) or "lines" (surface and bed)
VEL_NZ = 60           # vertical points of the contour grid (more: finer, but a bigger html)


def repo_path(p):
    return p if p is None or os.path.isabs(p) else os.path.join(REPO, p)


def extract(path):
    """Basal T(x), Zs(x) and, if available, velocities at the bed and surface of one state file.
    Velocities come from <state>_vel.npz (`ess velocity`) when it exists, else from the file itself."""
    s = load_state(path)
    diag = s.diag
    vpath = os.path.splitext(path)[0] + "_vel.npz"
    vdiag = load_state(vpath).diag if os.path.exists(vpath) else diag
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
    o = o_b = np.argsort(xb)
    out["xb"], out["Tb"] = xb[o], split[tname][np.searchsorted(gids[tname], gb)][o]
    if zname:
        gs = gids[zname][on_side(gids[zname])]
        xs = at(gs, gx, X)
        o = o_s = np.argsort(xs)
        out["xs"], out["zs"] = xs[o], split[zname][np.searchsorted(gids[zname], gs)][o]
    vel = {}
    for c in (1, 2, 3):
        try:
            vel[c] = vdiag[find_name(vdiag, f"velocity {c}")]
        except KeyError:
            pass
    if 1 in vel and 3 in vel:
        gb_s, gs_s = gb[o_b], (gs[o_s] if zname else None)
        out["ub"] = at(gb_s, *vel[1])
        out["wb"] = at(gb_s, *vel[3])
        if zname:
            out["us"] = at(gs_s, *vel[1])
            out["ws"] = at(gs_s, *vel[3])
            if 2 in vel:
                out["vs"] = at(gs_s, *vel[2])
        if VEL_PLOT == "contour":
            # x-z sections of u and w on the ice nodes of one side, gridded column by column
            xbb, zbb = xb[o_b], at(gb[o_b], gz, Z)
            for c, k in ((1, "U"), (3, "W")):
                gv, vv = vel[c]
                sv = on_side(gv)
                xv, zv, val = at(gv[sv], gx, X), at(gv[sv], gz, Z), vv[sv]
                ice = zv >= np.interp(xv, xbb, zbb) - 1e-6
                xg, zg, G = column_grid(xv[ice], zv[ice], val[ice], VEL_NZ)
                out[k] = (xg / 1e3, zg, np.round(G, 3))
    if "Tb_cold_max" in s.meta:
        out["Tb_cold_max"] = float(s.meta["Tb_cold_max"])
    return out


def main():
    bdir = repo_path(sys.argv[1] if len(sys.argv) > 1 else BRANCH)
    start = repo_path(sys.argv[2] if len(sys.argv) > 2 else START)
    files = ([start] if start and os.path.exists(start) else []) + \
        sorted(glob.glob(os.path.join(bdir, "points", "state_[0-9][0-9][0-9][0-9].npz")))
    if not files:
        raise SystemExit(f"no states found (start {start}, branch {bdir}/points/state_*.npz)")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    S = [extract(f) for f in files]
    print(f"{len(S)} states: Tsurf = " + ", ".join(f"{s['p']:.3f}" for s in S))
    nv = sum("ub" in s for s in S)
    print(f"velocities for {nv} of {len(S)} states" + ("" if nv else " (run `ess velocity` to add them)"))
    if VIEW == "overlay":
        return overlay(S, bdir)
    tb_rng = [min(s["Tb"].min() for s in S) - 0.2, max(273.3, max(s["Tb"].max() for s in S) + 0.1)]
    xmax = max(s["xb"].max() for s in S) / 1e3

    vrows = any("ub" in s for s in S)
    contour = VEL_PLOT == "contour" and any("U" in s for s in S)
    E = go.Scatter(x=[], y=[])
    crange = {}
    if contour:                               # one colour range for all states
        umax = max(np.nanmax(s["U"][2]) for s in S if "U" in s)
        umin = min(np.nanmin(s["U"][2]) for s in S if "U" in s)
        wlim = max(np.nanmax(np.abs(s["W"][2])) for s in S if "W" in s) or 1.0
        crange = {"U": dict(colorscale="Viridis", zmin=umin, zmax=umax, colorbar=dict(title="u (m/yr)", y=0.37, len=0.22)),
                  "W": dict(colorscale="RdBu_r", zmin=-wlim, zmax=wlim, colorbar=dict(title="w (m/yr)", y=0.11, len=0.22))}

    def line(x, y, colour, name, unit, dash=None):
        return go.Scatter(x=x / 1e3, y=y, mode="lines", line=dict(color=colour, dash=dash), name=name,
                          hovertemplate=f"x %{{x:.1f}} km<br>{name} %{{y:.4g}} {unit}<extra></extra>")

    def traces(s):
        tr = [go.Scatter(x=s["xb"] / 1e3, y=np.round(s["Tb"], 4), mode="lines", line=dict(color="#c0392b"),
                         name="T bed", hovertemplate="x %{x:.1f} km<br>T %{y:.3f} K<extra></extra>")]
        tr.append(go.Scatter(x=s["xs"] / 1e3, y=np.round(s["zs"], 2), mode="lines", line=dict(color="#2c3e50"),
                             name="Zs", hovertemplate="x %{x:.1f} km<br>Zs %{y:.1f} m<extra></extra>")
                  if "xs" in s else go.Scatter(x=[], y=[]))
        if vrows and contour:
            for k, nm in (("U", "u"), ("W", "w")):
                if k in s:
                    xg, zg, G = s[k]
                    tr.append(go.Contour(x=xg, y=zg, z=G, name=nm, connectgaps=False, ncontours=30,
                                         contours=dict(coloring="fill", showlines=False), line=dict(width=0),
                                         **crange[k],
                                         hovertemplate=f"x %{{x:.1f}} km<br>z %{{y:.0f}} m<br>{nm} %{{z:.4g}} m/yr"
                                                       "<extra></extra>"))
                else:
                    tr.append(E)
        elif vrows:
            tr.append(line(s["xs"], s["us"], "#1f77b4", "u surface", "m/yr") if "us" in s else E)
            tr.append(line(s["xb"], s["ub"], "#ff7f0e", "u bed", "m/yr") if "ub" in s else E)
            tr.append(line(s["xs"], s["ws"], "#2ca02c", "w surface", "m/yr") if "ws" in s else E)
            tr.append(line(s["xb"], s["wb"], "#9467bd", "w bed", "m/yr", "dot") if "wb" in s else E)
        return tr

    def title(s):
        extra = f", warmest cold-bed node {s['Tb_cold_max']:.3f} K" if "Tb_cold_max" in s else ""
        return f"{s['name']}: Tsurf = {s['p']:.4f} K{extra}"

    rows = 4 if vrows else 2
    titles = ["basal temperature (K)", "surface elevation (m)"]
    if contour:
        titles += ["along-flow velocity u (m/yr)", "vertical velocity w (m/yr)"]
    elif vrows:
        titles += ["along-flow velocity u (m/yr): surface, bed", "vertical velocity w (m/yr): surface, bed"]
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.06, subplot_titles=titles)
    where = ([1, 2, 3, 4] if contour else [1, 2, 3, 3, 4, 4])[:len(traces(S[0]))]
    for i, t in enumerate(traces(S[0])):
        fig.add_trace(t, where[i], 1)
    fig.add_hline(y=273.15, line=dict(dash="dot", color="grey"), row=1, col=1)
    frames, steps = [], []
    for k, s in enumerate(S):
        frames.append(go.Frame(name=str(k), data=traces(s), traces=list(range(len(where))),
                               layout=dict(title_text=title(s))))
        steps.append(dict(method="animate", label=f"{s['p']:.3f}",
                          args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0))]))
    fig.frames = frames
    for r in range(1, rows + 1):
        fig.update_xaxes(range=[0, xmax], row=r, col=1)
    fig.update_xaxes(title_text="x (km)", row=rows, col=1)
    if contour:
        zlo = min(s["U"][1][0] for s in S if "U" in s)
        zhi = max(s["U"][1][-1] for s in S if "U" in s)
        for r in (3, 4):
            fig.update_yaxes(range=[zlo, zhi], title_text="z (m)", row=r, col=1)
    elif vrows:
        def rng(keys):
            v = np.concatenate([s[k] for s in S for k in keys if k in s])
            pad = 0.05 * (v.max() - v.min() or 1.0)
            return [v.min() - pad, v.max() + pad]
        fig.update_yaxes(range=rng(["us", "ub"]), row=3, col=1)
        fig.update_yaxes(range=rng(["ws", "wb"]), row=4, col=1)
    fig.update_yaxes(range=tb_rng, row=1, col=1)
    zs_all = np.concatenate([s["zs"] for s in S if "zs" in s])
    fig.update_yaxes(range=[zs_all.min() - 50, zs_all.max() + 50], row=2, col=1)
    fig.update_layout(
        title=title(S[0]), height=400 * rows, template="plotly_white", showlegend=False, margin=dict(t=90, b=130),
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


def overlay(S, bdir):
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    from plotly.subplots import make_subplots

    n = len(S)
    colours = sample_colorscale(COLORSCALE, list(np.linspace(0, 1, n) if n > 1 else [0.0]))
    diff = MODE == "difference"
    ref = S[0]
    vrows = any("ub" in s for s in S)
    rows = 4 if vrows else 2
    ch = " change from the first state" if diff else ""
    titles = ["basal temperature" + ch + " (K)", "surface elevation" + ch + " (m)"]
    if vrows:
        titles += ["along-flow velocity u" + ch + " (m/yr): surface solid, bed dashed",
                   "vertical velocity w at the surface" + ch + " (m/yr)"]
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.06, shared_xaxes=True, subplot_titles=titles)

    def rel(s, xk, k):
        """field k of state s, minus the first state's (interpolated) in difference mode"""
        if not diff:
            return s[k]
        return s[k] - np.interp(s[xk], ref[xk], ref[k]) if k in ref else s[k] * np.nan
    for k, (s, c) in enumerate(zip(S, colours)):
        lab = f"{k}: {s['name']}<br>Tsurf {s['p']:.4f} K"
        tb = s["Tb"] - np.interp(s["xb"], ref["xb"], ref["Tb"]) if diff else s["Tb"]
        fig.add_trace(go.Scatter(x=s["xb"] / 1e3, y=tb, mode="lines", line=dict(color=c, width=1.5),
                                 showlegend=False,
                                 hovertemplate=lab + "<br>x %{x:.1f} km<br>%{y:.4f}<extra></extra>"), 1, 1)
        if "xs" in s:
            zs = s["zs"] - np.interp(s["xs"], ref["xs"], ref["zs"]) if diff else s["zs"]
            fig.add_trace(go.Scatter(x=s["xs"] / 1e3, y=zs, mode="lines", line=dict(color=c, width=1.5),
                                     showlegend=False,
                                     hovertemplate=lab + "<br>x %{x:.1f} km<br>%{y:.2f}<extra></extra>"), 2, 1)
        for xk, k, r, dash in (("xs", "us", 3, None), ("xb", "ub", 3, "dash"), ("xs", "ws", 4, None)):
            if vrows and k in s:
                fig.add_trace(go.Scatter(x=s[xk] / 1e3, y=rel(s, xk, k), mode="lines", showlegend=False,
                                         line=dict(color=c, width=1.5, dash=dash),
                                         hovertemplate=lab + f"<br>{k}<br>x %{{x:.1f}} km<br>%{{y:.4g}} m/yr"
                                                       "<extra></extra>"), r, 1)
    if not diff:
        fig.add_hline(y=273.15, line=dict(dash="dot", color="grey"), row=1, col=1)
    # colour bar: an invisible marker trace carrying the scale, ticks labelled with Tsurf
    ti = np.unique(np.round(np.linspace(0, n - 1, min(n, 8))).astype(int))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=COLORSCALE, cmin=0, cmax=max(n - 1, 1), color=[0], showscale=True,
                    colorbar=dict(title="Tsurf (K)<br>along the branch", tickvals=list(ti),
                                  ticktext=[f"{S[i]['p']:.3f}" for i in ti], len=0.9))), 1, 1)
    fig.update_xaxes(title_text="x (km)", row=rows, col=1)
    fig.update_layout(title=f"{os.path.basename(os.path.normpath(bdir))}: {n} states, "
                            f"Tsurf {min(s['p'] for s in S):.4f}-{max(s['p'] for s in S):.4f} K",
                      height=400 * rows, template="plotly_white", hovermode="closest")
    out = os.path.join(bdir, f"overlay_{MODE}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
