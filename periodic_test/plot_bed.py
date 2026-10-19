#!/usr/bin/env python3
"""
Basal temperature of the periodic-BC test cases on the 2D footprint, to see partition artefacts.

Copy the results to your laptop (only the small bed.npz files are needed), e.g.
    rsync -av --include='*/' --include='bed.npz' --include='summary.csv' --exclude='*' \\
        tethys:/path/to/ElmerSteadyStates/periodic_test/out/ periodic_test/out/
then press Run (settings below) or:
    python periodic_test/plot_bed.py periodic_test/out/np8

One page, a menu to pick the case. Rows (x along flow, y across):
  1  basal temperature
  2  basal temperature minus its across-flow mean   (exact: 0; partition artefacts show here)
  3  basal temperature minus the serial direct reference of the same physics
  4  nodal friction load minus its across-flow mean, relative (loads cases)
  5  partitions (colour = lowest partition holding the node)
Black dots in rows 2-5 mark the nodes shared between partitions, i.e. the partition interfaces.
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
CASES = None                      # list of case names, or None for every case with a bed.npz
SHOW_INTERFACES = True
SHOW = True


def grid(x, y, v):
    ux, uy = np.unique(np.round(x, 2)), np.unique(np.round(y, 2))
    G = np.full((len(uy), len(ux)), np.nan)
    G[np.searchsorted(uy, np.round(y, 2)), np.searchsorted(ux, np.round(x, 2))] = v
    return ux / 1e3, uy / 1e3, G


def across_mean_dev(x, v):
    d = np.array(v, float)
    xr = np.round(x, 2)
    for xv in np.unique(xr):
        m = xr == xv
        d[m] = v[m] - np.nanmean(v[m])
    return d


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DIR
    d = d if os.path.isabs(d) else os.path.join(REPO, d)
    files = sorted(glob.glob(os.path.join(d, "*", "bed.npz")))
    if CASES:
        files = [f for f in files if os.path.basename(os.path.dirname(f)) in CASES]
    if not files:
        raise SystemExit(f"no bed.npz under {d} (run the analysis on tethys, then copy them)")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    titles = ["basal temperature (K)", "minus its across-flow mean (K)",
              "minus the serial direct reference (K)", "friction load minus its across-flow mean (relative)",
              "partitions (lowest partition holding each node)"]
    rows = len(titles)
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.045, subplot_titles=titles)
    cb = lambda r, t: dict(title=t, len=0.8 / rows, y=1 - (r - 0.5) / rows)
    groups, names, heads = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        name = str(z["case"])
        x, y, T = z["x"], z["y"], z["T"]
        iface = z["nparts"] > 1
        tr = []

        def add(r, v, cs, title, sym=False, scale=1.0):
            ux, uy, G = grid(x, y, v / scale)
            lim = np.nanmax(np.abs(G)) if sym else None
            if sym and not (lim and np.isfinite(lim)):
                lim = 1e-12
            tr.append((go.Heatmap(x=ux, y=uy, z=G, colorscale=cs, zmin=-lim if sym else None,
                                  zmax=lim if sym else None, colorbar=cb(r, title),
                                  hovertemplate="x %{x:.2f} km<br>y %{y:.2f} km<br>%{z:.4g}<extra></extra>"), r))
            if SHOW_INTERFACES and iface.any() and r > 1:
                tr.append((go.Scatter(x=x[iface] / 1e3, y=y[iface] / 1e3, mode="markers", hoverinfo="skip",
                                      marker=dict(size=3, color="black")), r))

        add(1, T, "RdYlBu_r", "K")
        add(2, across_mean_dev(x, T), "RdBu_r", "K", sym=True)
        if "dT_ref" in z.files:
            add(3, z["dT_ref"], "RdBu_r", "K", sym=True)
        if "fric_load" in z.files and str(z["heat"]) == "loads":
            fl = z["fric_load"]
            add(4, across_mean_dev(x, fl), "PuOr_r", "rel.", sym=True, scale=np.nanmax(np.abs(fl)) or 1.0)
        add(5, z["owner"].astype(float), "Turbo", "partition")
        groups.append(tr)
        names.append(name)
        dev = np.nanmax(np.abs(across_mean_dev(x, T)))
        heads.append(f"{name}: {int(z['nfiles'])} partitions ({z['parts']}), bc {z['bc']}, heating {z['heat']}"
                     f"{' (halved at y=0)' if bool(z['halve']) else ''}, heat solver {z['ls_heat']}, "
                     f"Stokes {z['ls_stokes']} {z['element']}<br>max |T - across-flow mean| = {dev:.2e} K;  "
                     f"{z['status']}")
    vis = []
    for i, tr in enumerate(groups):
        for t, r in tr:
            t.visible = i == 0
            fig.add_trace(t, r, 1)
        vis.append(len(tr))
    buttons = []
    for i, name in enumerate(names):
        v = [False] * sum(vis)
        s = sum(vis[:i])
        v[s:s + vis[i]] = [True] * vis[i]
        buttons.append(dict(label=name, method="update", args=[{"visible": v}, {"title.text": heads[i]}]))
    for r in range(1, rows + 1):
        fig.update_yaxes(title_text="y (km)", row=r, col=1)
    fig.update_xaxes(title_text="x (km)", row=rows, col=1)
    fig.update_layout(title=dict(text=heads[0], font=dict(size=13)), height=330 * rows, template="plotly_white",
                      showlegend=False, margin=dict(t=150),
                      updatemenus=[dict(buttons=buttons, direction="down", x=0, xanchor="left", y=1.06,
                                        yanchor="bottom", showactive=True)])
    out = os.path.join(d, "bed_maps.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out} ({len(names)} cases)")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
