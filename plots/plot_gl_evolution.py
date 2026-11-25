#!/usr/bin/env python3
"""
The ice sheet's evolution along a grounding-line branch run (jobs/gl_branch.py) as one page (plotly): a slider over the
saved chunk states, the surface Zs, the ice base Zb and the bed along the flow, the grounding line marked, in two
panels (the whole flowline, and a window around the grounding line that covers all the states), and the grounding line
against the state number with the current state marked.

    python3 plots/plot_gl_evolution.py LABEL=DIR_OR_STATES [LABEL2=DIR2 ...] [--out gl_evolution.html] [--every N]

Each argument: a label and a branch's transient_chunks/points folder (or its state files / a glob), e.g.
    r125w=work/flowline_gl_fine/branch_r125w/transient_chunks/points r125c=work/flowline_gl_fine/branch_r125c/transient_chunks/points
The states of one label are in the order of their state numbers, the labels in the order given (give the older job
first). The slider text: label, the file's number (the branch's chunk), the grounding line. --every N: every N-th
state of each label (and the last).
"""
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sia_start import bed_profile, column_profile  # noqa: E402
from ess.spectrum import gl_position  # noqa: E402
from ess.statefile import load_state  # noqa: E402


def files_of(spec):
    if os.path.isdir(spec):
        fs = glob.glob(os.path.join(spec, "state_*.npz"))
    else:
        fs = glob.glob(spec) if any(c in spec for c in "*?[") else [spec]
    num = lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[-1]) if re.findall(r"(\d+)", os.path.basename(f)) else 0
    return sorted(fs, key=num)


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        raise SystemExit(__doc__)
    out, every = "gl_evolution.html", 1
    for k in ("--out", "--every"):
        if k in a:
            i = a.index(k)
            v = a[i + 1]
            del a[i:i + 2]
            if k == "--out":
                out = v
            else:
                every = max(1, int(v))
    items = []
    for arg in a:
        label, _, spec = arg.partition("=")
        if not spec:
            raise SystemExit(f"{arg}: expected LABEL=DIR_OR_STATES")
        fs = files_of(spec)
        keep = [f for i, f in enumerate(fs) if i % every == 0 or i == len(fs) - 1]
        items += [(label, f) for f in keep]
    if not items:
        raise SystemExit("no state files found")
    prof = []
    for label, f in items:
        s = load_state(f)
        x, zs = column_profile(s, "zs")
        xb, zb = column_profile(s, "zb")
        xr, zr = bed_profile(s)
        n = re.findall(r"(\d+)", os.path.basename(f))
        prof.append(dict(label=label, n=int(n[-1]) if n else 0, gl=gl_position(s.frozen, s.diag) / 1e3,
                         x=x / 1e3, zs=zs, xb=xb / 1e3, zb=zb, xr=xr / 1e3, zr=zr, file=f))
        print(f"{label:8s} {os.path.basename(f):16s} GL {prof[-1]['gl']:8.2f} km  ({len(x)} columns)")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    gls = np.array([p["gl"] for p in prof])
    wlo, whi = gls.min() - 60.0, gls.max() + 100.0
    zall = np.concatenate([p["zs"] for p in prof])
    zlo = min(float(np.min([p["zb"].min() for p in prof])), float(np.min([p["zr"].min() for p in prof]))) - 30
    zhi = float(zall.max()) * 1.05
    zwin = max(float(p["zs"][(p["x"] >= wlo) & (p["x"] <= whi)].max()) for p in prof if ((p["x"] >= wlo) & (p["x"] <= whi)).any()) * 1.15 + 20
    fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]], row_heights=[0.5, 0.5],
                        subplot_titles=("whole flowline", "around the grounding line", "grounding line against state"),
                        vertical_spacing=0.14, horizontal_spacing=0.09)
    p0 = prof[-1]
    traces_per = 6

    def frame_traces(p):
        t = []
        for col_row in ((1, 1), (2, 1)):
            r, c = col_row
            t.append(go.Scatter(x=p["x"], y=p["zs"], mode="lines", line=dict(color="#1b5fa8", width=2.2), name="surface",
                                showlegend=(r == 1), xaxis=f"x{1 if r == 1 else 2}", yaxis=f"y{1 if r == 1 else 2}"))
            t.append(go.Scatter(x=p["xb"], y=p["zb"], mode="lines", line=dict(color="#12857d", width=1.6), name="ice base",
                                showlegend=(r == 1), xaxis=f"x{1 if r == 1 else 2}", yaxis=f"y{1 if r == 1 else 2}"))
            t.append(go.Scatter(x=p["xr"], y=p["zr"], mode="lines", line=dict(color="#a89677", width=1.6), name="bed",
                                showlegend=(r == 1), fill="tozeroy" if False else None,
                                xaxis=f"x{1 if r == 1 else 2}", yaxis=f"y{1 if r == 1 else 2}"))
        return t
    # data traces: 6 per state (3 whole + 3 window), the slider toggles visibility; then the GL line markers and the GL-vs-state panel
    for k, p in enumerate(prof):
        for t in frame_traces(p):
            t.visible = (k == len(prof) - 1)
            fig.add_trace(t, row=1 if t.xaxis == "x" or t.xaxis == "x1" else 2, col=1)
    ng = len(prof) * traces_per
    # grounding-line verticals for the current state (one trace per state per panel)
    for k, p in enumerate(prof):
        for r in (1, 2):
            fig.add_trace(go.Scatter(x=[p["gl"], p["gl"]], y=[zlo, zhi], mode="lines", line=dict(color="#c2410c", width=1.5, dash="dash"),
                                     name="grounding line", showlegend=(r == 1 and k == len(prof) - 1),
                                     visible=(k == len(prof) - 1)), row=r, col=1)
    # the panel 3: grounding line against state index, with the current state marked
    idx = list(range(len(prof)))
    fig.add_trace(go.Scatter(x=idx, y=gls, mode="lines+markers", line=dict(color="#7a4fb0"), marker=dict(size=5),
                             name="grounding line (km)", showlegend=False,
                             text=[f"{p['label']} {p['n']}" for p in prof], hovertemplate="%{text}: %{y:.2f} km<extra></extra>"),
                  row=2, col=2)
    for k in idx:
        fig.add_trace(go.Scatter(x=[k], y=[gls[k]], mode="markers", marker=dict(size=11, color="#c2410c"), showlegend=False,
                                 hoverinfo="skip", visible=(k == idx[-1])), row=2, col=2)
    nt = len(fig.data)
    nm = nt - len(prof)           # first marker trace
    steps = []
    for k, p in enumerate(prof):
        vis = [False] * nt
        for j in range(traces_per):
            vis[k * traces_per + j] = True
        for r in (0, 1):
            vis[ng + k * 2 + r] = True
        vis[nm - 1] = True        # the grounding-line-against-state line
        vis[nm + k] = True
        steps.append(dict(method="update", args=[{"visible": vis}, {}],
                          label=f"{p['label']} {p['n']}: {p['gl']:.1f} km"))
    fig.update_layout(sliders=[dict(active=len(prof) - 1, steps=steps, currentvalue=dict(prefix="state: "), pad=dict(t=55))],
                      height=860, margin=dict(l=60, r=20, t=70, b=70), template="plotly_white",
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=1, xanchor="right"), title="Grounding-line branch: ice sheet evolution")
    fig.update_xaxes(title_text="distance from the divide (km)", row=1, col=1)
    fig.update_xaxes(title_text="distance from the divide (km)", range=[wlo, whi], row=2, col=1)
    fig.update_xaxes(title_text="state number", row=2, col=2)
    fig.update_yaxes(title_text="elevation (m)", range=[zlo, zhi], row=1, col=1)
    fig.update_yaxes(title_text="elevation (m)", range=[zlo, zwin], row=2, col=1)
    fig.update_yaxes(title_text="grounding line (km)", row=2, col=2)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"{len(prof)} states -> {out}")


if __name__ == "__main__":
    main()
