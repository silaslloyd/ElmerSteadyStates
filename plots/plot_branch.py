#!/usr/bin/env python3
"""
Bifurcation diagram of one or several continuation branches: block means of the state (and the mask
diagnostics, if present) against the parameter, one colour per branch, folds marked (◆), points with
unstable modes marked (✕) when stability was computed along the branch (`stability_every`).

Press Run in an IDE (settings below), or:
    python plots/plot_branch.py                                  # every branch under work/
    python plots/plot_branch.py work/land_margin/branch_flux work/land_margin_d02/branch_Tsurf
Needs numpy and plotly. Linear axes.
"""
import csv
import glob
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from ess.statefile import load_state  # noqa: E402

# ---- settings for pressing Run (paths relative to the repository)
BRANCHES = None      # list of branch directories; None -> every */branch_*/ with a branch.csv under work/
EXCLUDE = []         # substrings: branches whose path contains one of these are left out (e.g. "land_flowline")
SHOW = True
OUT = "work/branches.html"
COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
           "#7f7f7f", "#bcbd22"]


def repo_path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def label(bdir):
    """work/<case>/branch_<name> -> '<case> / <name>'"""
    b = os.path.normpath(bdir)
    case, name = os.path.basename(os.path.dirname(b)), os.path.basename(b).replace("branch_", "")
    return f"{case} / {name}"


def read_branch(bdir):
    with open(os.path.join(bdir, "branch.csv")) as f:
        rows = list(csv.DictReader(f))
    p = np.array([float(r["p"]) for r in rows])
    fold = np.array([str(r.get("fold", "0")) == "1" for r in rows])
    nun = np.array([float(r["n_unstable"]) if r.get("n_unstable") not in (None, "") else np.nan for r in rows])
    means = {}

    def put(k, i, v):
        means.setdefault(k, np.full(len(rows), np.nan))[i] = v

    for i in range(len(rows)):
        sp = os.path.join(bdir, "points", f"state_{i:04d}.npz")
        if os.path.exists(sp):
            s = load_state(sp)
            for k, v in s.layout.split(s.x).items():
                put(f"mean {k}", i, float(np.mean(v)))
            if "Tb_cold_mean" in s.meta:
                put("Tb_cold_mean (basal, cold bed)", i, float(s.meta["Tb_cold_mean"]))
                put("Tb_cold_max (warmest cold-bed node)", i, float(s.meta["Tb_cold_max"]))
        else:
            pt = os.path.join(bdir, "points", f"pt_{i:04d}.npz")
            if os.path.exists(pt):
                with np.load(pt) as z:
                    put("mean (scaled)", i, float(np.mean(z["y"])))
    return dict(p=p, fold=fold, nun=nun, means=means)


def main():
    if len(sys.argv) > 1:
        dirs = [repo_path(a) for a in sys.argv[1:]]
    elif BRANCHES:
        dirs = [repo_path(b) for b in BRANCHES]
    else:
        dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(REPO, "work", "*", "branch_*", "branch.csv")))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "branch.csv")) and not any(e in d for e in EXCLUDE)]
    if not dirs:
        raise SystemExit("no branch.csv found (looked under work/*/branch_*/)")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    B = []
    for d in dirs:
        try:
            B.append((d, read_branch(d)))
        except Exception as e:                  # an empty or half-written branch
            print(f"skipping {d}: {e}")
    names = []
    for _, b in B:                               # panels: every quantity that some branch has
        names += [k for k in b["means"] if k not in names]
    print(f"{len(B)} branches: " + ", ".join(label(d) for d, _ in B))

    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=names)
    for ib, (d, b) in enumerate(B):
        c = COLOURS[ib % len(COLOURS)]
        lab = label(d)
        for j, k in enumerate(names, 1):
            if k not in b["means"]:
                continue
            y = b["means"][k]
            first = j == 1
            fig.add_trace(go.Scatter(x=b["p"], y=y, mode="lines+markers", line=dict(color=c, width=1.5),
                                     marker=dict(color=c, size=5), name=lab, legendgroup=lab, showlegend=first,
                                     text=[f"{lab}, point {i}" for i in range(len(y))],
                                     hovertemplate="%{text}<br>p %{x:.5g}<br>%{y:.6g}<extra></extra>"), j, 1)
            if b["fold"].any():
                fig.add_trace(go.Scatter(x=b["p"][b["fold"]], y=y[b["fold"]], mode="markers",
                                         marker=dict(symbol="diamond", size=11, color=c, line=dict(color="black", width=1)),
                                         legendgroup=lab, showlegend=False, hoverinfo="skip"), j, 1)
            un = b["nun"] > 0
            if un.any():
                fig.add_trace(go.Scatter(x=b["p"][un], y=y[un], mode="markers",
                                         marker=dict(symbol="x", size=9, color="black"),
                                         legendgroup=lab, showlegend=False, hoverinfo="skip"), j, 1)
    fig.update_xaxes(title_text="parameter", row=len(names), col=1)
    fig.update_layout(title="branches  (◆ fold, ✕ unstable point where stability was computed)",
                      height=300 * len(names) + 120, template="plotly_white",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    out = repo_path(OUT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print("wrote", out)
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
