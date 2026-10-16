#!/usr/bin/env python3
"""
Bifurcation diagram of a continuation branch: block means of the state (and the mask diagnostics,
if present) against the parameter, with folds marked and points coloured by stability when
`stability_every` was set.

Press Run in an IDE (settings below), or:  python plots/plot_branch.py work/land_flowline/branch_Tsurf
Needs numpy and plotly. Linear axes.
"""
import csv
import glob
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ess.statefile import load_state  # noqa: E402

# ---- settings for pressing Run
BRANCH = None        # branch directory; None -> the newest */branch_*/branch.csv under work/
SHOW = True


def main():
    bdir = sys.argv[1] if len(sys.argv) > 1 else BRANCH
    if bdir is None:
        files = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "work", "**", "branch.csv"), recursive=True),
                       key=os.path.getmtime)
        if not files:
            raise SystemExit("no branch.csv under work/")
        bdir = os.path.dirname(files[-1])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    with open(os.path.join(bdir, "branch.csv")) as f:
        rows = list(csv.DictReader(f))
    p = np.array([float(r["p"]) for r in rows])
    fold = np.array([str(r.get("fold", "0")) == "1" for r in rows])
    nun = np.array([float(r["n_unstable"]) if r.get("n_unstable") not in (None, "") else np.nan for r in rows])

    # block means from the saved physical states (Elmer runs) or the scaled points (toy)
    means = {}
    for i in range(len(rows)):
        sp = os.path.join(bdir, "points", f"state_{i:04d}.npz")
        if os.path.exists(sp):
            s = load_state(sp)
            for k, v in s.layout.split(s.x).items():
                means.setdefault(f"mean {k}", []).append(float(np.mean(v)))
            if "Tb_cold_mean" in s.meta:
                means.setdefault("Tb_cold_mean (basal, cold bed)", []).append(float(s.meta["Tb_cold_mean"]))
                means.setdefault("Tb_cold_max", []).append(float(s.meta["Tb_cold_max"]))
        else:
            with np.load(os.path.join(bdir, "points", f"pt_{i:04d}.npz")) as z:
                means.setdefault("first entry (scaled)", []).append(float(z["y"][0]))
                means.setdefault("mean (scaled)", []).append(float(np.mean(z["y"])))
    names = [k for k, v in means.items() if len(v) == len(rows)]
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=names)
    for j, k in enumerate(names, 1):
        y = np.array(means[k])
        fig.add_trace(go.Scatter(x=p, y=y, mode="lines", line=dict(color="#888"), showlegend=False), j, 1)
        col = np.where(np.isnan(nun), "#1f77b4", np.where(nun > 0, "#d62728", "#2ca02c"))
        fig.add_trace(go.Scatter(x=p, y=y, mode="markers", marker=dict(color=col, size=7),
                                 text=[f"point {i}, unstable modes: {'?' if np.isnan(u) else int(u)}"
                                       for i, u in enumerate(nun)],
                                 hovertemplate="p %{x:.5g}<br>%{y:.6g}<br>%{text}<extra></extra>",
                                 showlegend=False), j, 1)
        if fold.any():
            fig.add_trace(go.Scatter(x=p[fold], y=y[fold], mode="markers", marker=dict(symbol="x", size=12,
                          color="black"), name="fold", showlegend=(j == 1)), j, 1)
    fig.update_xaxes(title_text="parameter", row=len(names), col=1)
    fig.update_layout(title=f"{bdir}  (green: stable, red: unstable, blue: not computed)",
                      height=300 * len(names) + 100, template="plotly_white")
    out = os.path.join(bdir, "branch.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print("wrote", out)
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
