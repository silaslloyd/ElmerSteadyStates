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
from plot_branch import read_stability  # noqa: E402

# ---- settings for pressing Run (paths relative to the repository)
BRANCH = "work/land_flowline/branch_cold"              # branch directory
START = "work/land_flowline/steady_p268.15_cold.npz"   # starting state (None: only the branch points)
Y_SIDE = "min"        # the flowline mesh is one element wide: nodes at y = min (or "max")
SHOW = True

VIEW = "slider"       # "slider": one state at a time; "overlay": all states, coloured along the branch
MODE = "absolute"     # overlay only: "absolute", or "difference" (change from the first state)
COLORSCALE = "Viridis"
VEL_PLOT = "contour"  # slider only: "contour" (filled x-z sections of T, u, w in the ice, surface drawn on
                      # them) or "lines" (surface elevation panel; u, w at the surface and bed)
VEL_NZ = 60           # vertical points of the contour grids (more: finer, but a bigger html)
BRANCH_Y = "Tb_mean"  # slider only, top panel (the branch, with the current state marked): y-axis quantity,
                      # "Tb_mean" (mean basal temperature), "Tb_max" (warmest bed node) or "Zs_mean"
# stability type of the leading mode (from `ess stability --name B --point K`, or n_unstable in branch.csv)
STAB_COLOURS = {"stable": "#1f77b4", "stable, oscillatory": "#17becf",
                "unstable": "#d62728", "unstable, oscillatory": "#ff7f0e"}


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
    base = os.path.basename(path)
    out["k"] = int(base[6:10]) if base.startswith("state_") and base[6:10].isdigit() else None
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
    xbb, zbb = xb[o_b], at(gb[o_b], gz, Z)          # the bed, for keeping sections in the ice
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
            for c, k in ((1, "U"), (3, "W")):
                gv, vv = vel[c]
                sv = on_side(gv)
                xv, zv, val = at(gv[sv], gx, X), at(gv[sv], gz, Z), vv[sv]
                ice = zv >= np.interp(xv, xbb, zbb) - 1e-6
                xg, zg, G = column_grid(xv[ice], zv[ice], val[ice], VEL_NZ)
                out[k] = (xg / 1e3, zg, np.round(G, 3))
    if VEL_PLOT == "contour":
        # temperature section in the ice (the bedrock below the bed nodes is left out)
        gt = gids[tname][on_side(gids[tname])]
        xt, zt = at(gt, gx, X), at(gt, gz, Z)
        tv = split[tname][np.searchsorted(gids[tname], gt)]
        ice = zt >= np.interp(xt, xbb, zbb) - 1e-6
        xg, zg, G = column_grid(xt[ice], zt[ice], tv[ice], VEL_NZ)
        out["TS"] = (xg / 1e3, zg, np.round(G, 3))
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
    sections = VEL_PLOT == "contour"
    contour = sections and any("U" in s for s in S)     # velocity sections
    E = go.Scatter(x=[], y=[])
    crange = {}
    if sections:                              # one colour range for all states
        crange["TS"] = dict(colorscale="RdYlBu_r", zmin=min(np.nanmin(s["TS"][2]) for s in S),
                            zmax=max(np.nanmax(s["TS"][2]) for s in S), colorbar=dict(title="T (K)"))
    if contour:
        umax = max(np.nanmax(s["U"][2]) for s in S if "U" in s)
        umin = min(np.nanmin(s["U"][2]) for s in S if "U" in s)
        wlim = max(np.nanmax(np.abs(s["W"][2])) for s in S if "W" in s) or 1.0
        crange["U"] = dict(colorscale="Viridis", zmin=umin, zmax=umax, colorbar=dict(title="u (m/yr)"))
        crange["W"] = dict(colorscale="RdBu_r", zmin=-wlim, zmax=wlim, colorbar=dict(title="w (m/yr)"))

    def line(x, y, colour, name, unit, dash=None):
        return go.Scatter(x=x / 1e3, y=y, mode="lines", line=dict(color=colour, dash=dash), name=name,
                          hovertemplate=f"x %{{x:.1f}} km<br>{name} %{{y:.4g}} {unit}<extra></extra>")

    def section(s, k, nm, unit):
        if k not in s:
            return E
        xg, zg, G = s[k]
        return go.Contour(x=xg, y=zg, z=G, name=nm, connectgaps=False, ncontours=30,
                          contours=dict(coloring="fill", showlines=False), line=dict(width=0), **crange[k],
                          hovertemplate=f"x %{{x:.1f}} km<br>z %{{y:.0f}} m<br>{nm} %{{z:.4g}} {unit}<extra></extra>")

    def surface(s):
        return (go.Scatter(x=s["xs"] / 1e3, y=s["zs"], mode="lines", line=dict(color="black", width=1.5),
                           name="Zs", hovertemplate="x %{x:.1f} km<br>Zs %{y:.1f} m<extra></extra>")
                if "xs" in s else E)

    def traces(s):
        tr = [go.Scatter(x=s["xb"] / 1e3, y=np.round(s["Tb"], 4), mode="lines", line=dict(color="#c0392b"),
                         name="T bed", hovertemplate="x %{x:.1f} km<br>T %{y:.3f} K<extra></extra>")]
        if sections:                               # T section, then u and w, each with the surface on it
            tr += [section(s, "TS", "T", "K"), surface(s)]
            if contour:
                tr += [section(s, "U", "u", "m/yr"), surface(s), section(s, "W", "w", "m/yr"), surface(s)]
            return tr
        tr.append(go.Scatter(x=s["xs"] / 1e3, y=np.round(s["zs"], 2), mode="lines", line=dict(color="#2c3e50"),
                             name="Zs", hovertemplate="x %{x:.1f} km<br>Zs %{y:.1f} m<extra></extra>")
                  if "xs" in s else go.Scatter(x=[], y=[]))
        if vrows:
            tr.append(line(s["xs"], s["us"], "#1f77b4", "u surface", "m/yr") if "us" in s else E)
            tr.append(line(s["xb"], s["ub"], "#ff7f0e", "u bed", "m/yr") if "ub" in s else E)
            tr.append(line(s["xs"], s["ws"], "#2ca02c", "w surface", "m/yr") if "ws" in s else E)
            tr.append(line(s["xb"], s["wb"], "#9467bd", "w bed", "m/yr", "dot") if "wb" in s else E)
        return tr

    def title(s):
        extra = f", warmest cold-bed node {s['Tb_cold_max']:.3f} K" if "Tb_cold_max" in s else ""
        return f"{s['name']}: Tsurf = {s['p']:.4f} K{extra}"

    # ---- top panel: the branch, coloured by stability type, with the current state marked
    stab_cat = branch_stability(bdir, S)

    def yval(s):
        return {"Tb_mean": np.mean(s["Tb"]), "Tb_max": np.max(s["Tb"]),
                "Zs_mean": np.mean(s["zs"]) if "zs" in s else np.nan}[BRANCH_Y]
    ylab = {"Tb_mean": "mean basal T (K)", "Tb_max": "warmest bed node (K)", "Zs_mean": "mean Zs (m)"}[BRANCH_Y]
    pts = sorted((s["k"], s["p"], yval(s)) for s in S if s["k"] is not None)

    o = 1                                          # the state panels start in row 2
    titles = ["branch: " + ", ".join(f"<span style='color:{c}'>{n}</span>" for n, c in STAB_COLOURS.items())
              + ", grey not computed (between two computed points of the same type the type is assumed)",
              "basal temperature (K)"]
    if sections:
        titles += ["temperature (K)"] + (["along-flow velocity u (m/yr)", "vertical velocity w (m/yr)"]
                                         if contour else [])
        where = [1, 2, 2] + ([3, 3, 4, 4] if contour else [])
        sec_rows = {"TS": 2 + o, "U": 3 + o, "W": 4 + o}
    else:
        titles += ["surface elevation (m)"]
        if vrows:
            titles += ["along-flow velocity u (m/yr): surface, bed", "vertical velocity w (m/yr): surface, bed"]
        where = [1, 2, 3, 3, 4, 4]
        sec_rows = {}
    rows = len(titles)
    for k_, r_ in sec_rows.items():                # colour bars beside their rows
        if k_ in crange:
            crange[k_]["colorbar"].update(y=1 - (r_ - 0.5) / rows, len=0.8 / rows)
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.05, subplot_titles=titles)
    where = [r + o for r in where[:len(traces(S[0]))]]

    def marker(s):
        return go.Scatter(x=[s["p"]], y=[yval(s)], mode="markers", name="current state",
                          marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="black", width=3)),
                          hovertemplate=f"{s['name']}<br>Tsurf %{{x:.4f}}<br>%{{y:.4g}}<extra></extra>")

    for i, t in enumerate(traces(S[0])):           # traces 0..n-1: the state panels, then the marker
        fig.add_trace(t, where[i], 1)
    fig.add_trace(marker(S[0]), 1, 1)
    nanim = len(where) + 1
    if pts:                                        # static: the branch and its stability colouring
        kk, pp, yy = map(np.array, zip(*pts))
        fig.add_trace(go.Scatter(x=pp, y=yy, mode="lines+markers", line=dict(color="#bbbbbb", width=2),
                                 marker=dict(size=4, color="#bbbbbb"), text=[f"point {k}" for k in kk],
                                 hovertemplate="%{text}<br>Tsurf %{x:.4f}<br>%{y:.4g}<extra></extra>"), 1, 1)
        seg = segment_colours(kk, stab_cat)
        i = 0
        while i < len(kk) - 1:                     # runs of equally coloured segments
            if seg[i] is None:
                i += 1
                continue
            j = i
            while j < len(kk) - 1 and seg[j] == seg[i]:
                j += 1
            fig.add_trace(go.Scatter(x=pp[i:j + 1], y=yy[i:j + 1], mode="lines", hoverinfo="skip",
                                     line=dict(color=STAB_COLOURS[seg[i]], width=5)), 1, 1)
            i = j
        ck = [i for i, k in enumerate(kk) if k in stab_cat]
        if ck:
            fig.add_trace(go.Scatter(
                x=pp[ck], y=yy[ck], mode="markers",
                marker=dict(size=11, color=[STAB_COLOURS[stab_cat[kk[i]][0]] for i in ck],
                            line=dict(color="black", width=1)),
                text=[f"point {kk[i]}: {stab_cat[kk[i]][0]}<br>{stab_cat[kk[i]][1]}" for i in ck],
                hovertemplate="%{text}<br>Tsurf %{x:.4f}<extra></extra>"), 1, 1)
    fig.update_xaxes(title_text="Tsurf (K)", row=1, col=1)
    fig.update_yaxes(title_text=ylab, row=1, col=1)
    fig.add_hline(y=273.15, line=dict(dash="dot", color="grey"), row=1 + o, col=1)
    frames, steps = [], []
    for k, s in enumerate(S):
        frames.append(go.Frame(name=str(k), data=traces(s) + [marker(s)], traces=list(range(nanim)),
                               layout=dict(title_text=title(s))))
        steps.append(dict(method="animate", label=f"{s['p']:.3f}",
                          args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0))]))
    fig.frames = frames
    for r in range(1 + o, rows + 1):
        fig.update_xaxes(range=[0, xmax], row=r, col=1)
    fig.update_xaxes(title_text="x (km)", row=rows, col=1)
    if sections:
        zlo = min(s["TS"][1][0] for s in S)
        zhi = max(max(s["TS"][1][-1], np.max(s.get("zs", [-np.inf]))) for s in S)
        for r in sec_rows.values():
            if r <= rows:
                fig.update_yaxes(range=[zlo, zhi + 0.02 * (zhi - zlo)], title_text="z (m)", row=r, col=1)
    elif vrows:
        def rng(keys):
            v = np.concatenate([s[k] for s in S for k in keys if k in s])
            pad = 0.05 * (v.max() - v.min() or 1.0)
            return [v.min() - pad, v.max() + pad]
        fig.update_yaxes(range=rng(["us", "ub"]), row=3 + o, col=1)
        fig.update_yaxes(range=rng(["ws", "wb"]), row=4 + o, col=1)
    fig.update_yaxes(range=tb_rng, row=1 + o, col=1)
    if not sections:
        zs_all = np.concatenate([s["zs"] for s in S if "zs" in s])
        fig.update_yaxes(range=[zs_all.min() - 50, zs_all.max() + 50], row=2 + o, col=1)
    fig.update_layout(
        title=title(S[0]), height=380 * rows, template="plotly_white", showlegend=False, margin=dict(t=90, b=130),
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


def branch_stability(bdir, S):
    """{branch point: (type, details)} from the `ess stability` runs on the branch's points, else from
    n_unstable in branch.csv (stability_every; real/complex unknown there)."""
    cat = {}
    for k, st in read_stability(bdir).items():
        if not st["conv"]:
            continue
        kind = ("unstable" if st["sigma"] > 0 else "stable") + (", oscillatory" if st["complex"] else "")
        det = f"|mu| {st['absmu']:.5f}, sigma {st['sigma']:+.3e}/yr" + \
            (f", period {st['period'] / 1e3:.3g} kyr" if st["complex"] else "")
        cat[k] = (kind, det)
    f = os.path.join(bdir, "branch.csv")
    if os.path.exists(f):
        import csv
        with open(f) as fh:
            for k, r in enumerate(csv.DictReader(fh)):
                v = r.get("n_unstable")
                if k not in cat and v not in (None, ""):
                    cat[k] = ("unstable" if float(v) > 0 else "stable", f"stability_every: {int(float(v))} unstable")
    return cat


def segment_colours(kk, cat):
    """Colour of the segment from branch point kk[i] to kk[i+1]: the type of the computed points on either
    side when they agree (assumed constant between them), else None (grey)."""
    seg = [None] * (len(kk) - 1)
    ck = [k for k in kk if k in cat]
    for a, b in zip(ck[:-1], ck[1:]):
        if cat[a][0] == cat[b][0]:
            for i in range(len(kk) - 1):
                if a <= kk[i] and kk[i + 1] <= b:
                    seg[i] = cat[a][0]
    return seg


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
