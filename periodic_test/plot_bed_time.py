#!/usr/bin/env python3
"""
Basal temperature of the transient periodic-BC test cases (T_*) through time, on the 2D footprint, to see
partition artefacts appear and grow.

Copy the results to your laptop (only the small npz files are needed), e.g.
    rsync -av --include='*/' --include='bed*.npz' --include='summary.csv' --exclude='*' \\
        tethys:/path/to/ElmerSteadyStates/periodic_test/out/ periodic_test/out/
then press Run (settings below) or:
    python periodic_test/plot_bed_time.py periodic_test/out/np8 [CASE ...]

Writes one page per case, <DIR>/bed_time_<case>.html:
  row 1  growth: max over the bed of |T - its across-flow mean| against time, every T case (dotted:
         serial; exact: 0, so a partition artefact shows as a line rising above the serial reference)
and the case's maps, with a time slider:
  row 2  basal temperature
  row 3  basal temperature minus its across-flow mean
  row 4  basal temperature minus the serial reference at the same time
  row 5  partitions; black dots in rows 3-5 mark nodes shared between partitions
Needs numpy and plotly.
"""
import glob
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ---- settings for pressing Run
DIR = "periodic_test/out/np8"     # one out/np<N> folder
CASES = None                      # list of case names to make pages for; None: every T case
SHOW_INTERFACES = True
SHOW = True


def grid(x, y, v):
    ux, uy = np.unique(np.round(x, 2)), np.unique(np.round(y, 2))
    G = np.full((len(uy), len(ux)), np.nan)
    G[np.searchsorted(uy, np.round(y, 2)), np.searchsorted(ux, np.round(x, 2))] = v
    return ux / 1e3, uy / 1e3, G


def across_dev(x, v):
    """v minus its mean over y at each x (works on [time, node] arrays)"""
    v = np.atleast_2d(v)
    d = np.array(v, float)
    xr = np.round(x, 2)
    for xv in np.unique(xr):
        m = xr == xv
        d[:, m] = v[:, m] - np.nanmean(v[:, m], axis=1, keepdims=True)
    return d


def page(C, k0, d):
    """one html page: growth curves of every case (row 1), case k0's maps with a time slider"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    titles = ["growth of the across-flow variation of basal T (K), every case (this case thick)",
              "basal temperature (K)", "minus its across-flow mean (K)",
              "minus the serial reference at the same time (K)", "partitions"]
    rows = len(titles)
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.045, subplot_titles=titles,
                        row_heights=[0.22, 0.2, 0.2, 0.2, 0.18])
    cb = lambda r, t: dict(title=t, len=0.75 / rows, y=1 - (r - 0.5) / rows)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
    c = C[k0]
    x, y = c["x"], c["y"]

    def maps(it):
        tr = []
        ux, uy, G = grid(x, y, c["T"][it])
        tr.append(go.Heatmap(x=ux, y=uy, z=G, colorscale="RdYlBu_r", zmin=float(np.nanmin(c["T"])),
                             zmax=float(np.nanmax(c["T"])), colorbar=cb(2, "K"),
                             hovertemplate="x %{x:.2f} km<br>y %{y:.2f} km<br>%{z:.4f} K<extra></extra>"))
        lim = float(np.nanmax(np.abs(c["dev"]))) or 1e-12
        ux, uy, G = grid(x, y, c["dev"][it])
        tr.append(go.Heatmap(x=ux, y=uy, z=G, colorscale="RdBu_r", zmin=-lim, zmax=lim, colorbar=cb(3, "K"),
                             hovertemplate="x %{x:.2f} km<br>y %{y:.2f} km<br>%{z:.3e} K<extra></extra>"))
        if "dT_ref" in c and np.isfinite(c["dT_ref"]).any():
            lim = float(np.nanmax(np.abs(c["dT_ref"]))) or 1e-12
            ux, uy, G = grid(x, y, c["dT_ref"][it])
        else:
            G = G * np.nan
        tr.append(go.Heatmap(x=ux, y=uy, z=G, colorscale="RdBu_r", zmin=-lim, zmax=lim, colorbar=cb(4, "K"),
                             hovertemplate="x %{x:.2f} km<br>y %{y:.2f} km<br>%{z:.3e} K<extra></extra>"))
        tr.append(go.Scatter(x=[c["time"][it] / 1e3], y=[max(c["growth"][it], 1e-16)], mode="markers",
                             marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
                             hoverinfo="skip", showlegend=False))
        return tr

    nlast = len(c["time"]) - 1
    for t, r in zip(maps(nlast), (2, 3, 4, 1)):            # traces 0-3: animated
        fig.add_trace(t, r, 1)
    for i, o in enumerate(C):                              # growth curves
        fig.add_trace(go.Scatter(x=o["time"] / 1e3, y=np.maximum(o["growth"], 1e-16), mode="lines",
                                 name=o["name"], line=dict(color=palette[i % len(palette)], width=4 if i == k0 else 1.5,
                                                           dash="dot" if int(o["nfiles"]) == 1 else None),
                                 hovertemplate=o["name"] + "<br>t %{x:.2f} kyr<br>%{y:.2e} K<extra></extra>"), 1, 1)
    ux, uy, G = grid(x, y, c["owner"].astype(float))
    fig.add_trace(go.Heatmap(x=ux, y=uy, z=G, colorscale="Turbo", colorbar=cb(5, "partition"),
                             hovertemplate="partition %{z:.0f}<extra></extra>"), 5, 1)
    iface = c["nparts"] > 1
    if SHOW_INTERFACES and iface.any():
        for r in (3, 4, 5):
            fig.add_trace(go.Scatter(x=x[iface] / 1e3, y=y[iface] / 1e3, mode="markers", hoverinfo="skip",
                                     marker=dict(size=3, color="black"), showlegend=False), r, 1)
    frames, steps = [], []
    for it, t in enumerate(c["time"]):
        frames.append(go.Frame(name=str(it), data=maps(it), traces=[0, 1, 2, 3]))
        steps.append(dict(method="animate", label=f"{t / 1e3:.1f}",
                          args=[[str(it)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                transition=dict(duration=0))]))
    fig.frames = frames
    fig.update_yaxes(type="log", title_text="max |T - mean| (K)", row=1, col=1)
    fig.update_xaxes(title_text="time (kyr)", row=1, col=1)
    for r in range(2, rows + 1):
        fig.update_yaxes(title_text="y (km)", row=r, col=1)
    fig.update_xaxes(title_text="x (km)", row=rows, col=1)
    head = (f"{c['name']}: {int(c['nfiles'])} partitions ({c['parts']}), heating {c['heat']}"
            f"{' (halved at y=0)' if bool(c['halve']) else ''}, heat solver {c['ls_heat']}, Stokes {c['ls_stokes']}"
            f"<br>final max |T - across-flow mean| = {c['growth'][-1]:.2e} K;  {c['status']}")
    fig.update_layout(
        title=dict(text=head, font=dict(size=13)), height=300 * rows, template="plotly_white",
        legend=dict(x=1.08, y=1, yanchor="top", font=dict(size=10)), margin=dict(t=110, b=120),
        sliders=[dict(active=nlast, steps=steps, x=0.05, len=0.9, y=0, yanchor="top",
                      currentvalue=dict(prefix="t (kyr) = "), pad=dict(t=40))])
    out = os.path.join(d, f"bed_time_{c['name']}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DIR
    d = d if os.path.isabs(d) else os.path.join(REPO, d)
    want = sys.argv[2:] or CASES
    files = sorted(glob.glob(os.path.join(d, "*", "bed_time.npz")))
    if not files:
        raise SystemExit(f"no bed_time.npz under {d} (run the T cases and the analysis on tethys, then copy them)")
    C = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        c = {k: z[k] for k in z.files}
        c["name"] = str(z["case"])
        c["dev"] = across_dev(c["x"], c["T"])
        c["growth"] = np.nanmax(np.abs(c["dev"]), axis=1)
        C.append(c)
    print("cases: " + ", ".join(c["name"] for c in C))
    missing = [w for w in (want or []) if w not in [c["name"] for c in C]]
    if missing:
        print("no bed_time.npz (yet) for: " + ", ".join(missing))
    pages = [i for i, c in enumerate(C) if not want or c["name"] in want]
    if not pages:
        raise SystemExit("nothing to plot")
    for k in pages:
        out = page(C, k, d)
        print(f"wrote {out}")
        if SHOW and k == pages[0]:
            webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
