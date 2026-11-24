#!/usr/bin/env python3
"""The grounding line of a jobs/gl_branch.py run against model time, from the job's slurm log: one point per chunk
(1000 yr), the rms change of the chunk (right axis, log), the batch summaries (the BRANCH lines: mass balance), and the
start / reference grounding lines as lines. A page with plotly (hover for the numbers).

    python3 plots/plot_gl_branch.py slurms/gl_branch_32304.out [--ref 787.57] [--chunk-yr 1000] [--out gl_branch.html]

The log's lines used:
    run: chunk N/M  t = T  rms change R
    run: contact grounding line A -> B km          (the one after a chunk line: the grounding line at its end)
    pipeline: BRANCH r125: ... start state ...: grounding line S km (source ...)
    pipeline: BRANCH r125: K chunks (...): rms change R, grounding line G km (...), mass balance MB
"""
import os
import re
import sys


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        raise SystemExit(__doc__)
    log = a.pop(0)
    opt = {"--ref": "787.57", "--chunk-yr": "1000", "--out": "gl_branch.html"}
    while a:
        k = a.pop(0)
        if k not in opt or not a:
            raise SystemExit(f"unknown option {k}\n" + __doc__)
        opt[k] = a.pop(0)
    ref, dy, out = float(opt["--ref"]), float(opt["--chunk-yr"]), opt["--out"]

    start = None
    chunks = []                       # (model time yr, grounding line km, rms change)
    batches = []                      # (chunks done, GL km, rms, mass balance)
    pend = None
    n = 0
    for ln in open(log, errors="replace"):
        m = re.search(r"start state on the .* grounding line ([\d.]+) km", ln)
        if m:
            start = float(m.group(1))
        m = re.search(r"run: chunk (\d+)/\d+ +t = ([\d.eE+-]+) +rms change ([\d.eE+-]+)", ln)
        if m:
            pend = float(m.group(3))
            continue
        m = re.search(r"run: contact grounding line ([\d.]+) -> ([\d.]+) km", ln)
        if m and pend is not None:
            n += 1
            chunks.append((n * dy, float(m.group(2)), pend))
            pend = None
            continue
        m = re.search(r"BRANCH \S+ (\d+) chunks .*rms change ([\d.eE+-]+), grounding line ([\d.]+) km.*mass balance ([\d.eE+-]+)", ln)
        if m:
            batches.append((int(m.group(1)), float(m.group(3)), float(m.group(2)), float(m.group(4))))
    if not chunks:
        raise SystemExit(f"no chunk lines in {log}")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    t = [c[0] for c in chunks]
    t0 = ([0.0] if start is not None else []) + t
    gl = ([start] if start is not None else []) + [c[1] for c in chunks]
    fig.add_trace(go.Scatter(x=t0, y=gl, mode="lines+markers", name="grounding line (km)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=[c[2] for c in chunks], mode="lines+markers", name="rms change of the chunk",
                             line=dict(dash="dot")), secondary_y=True)
    if batches:
        fig.add_trace(go.Scatter(x=[b[0] * dy for b in batches], y=[b[1] for b in batches], mode="markers",
                                 marker=dict(size=12, symbol="diamond"), name="batch end",
                                 text=[f"mass balance {b[3]:.4f}, rms {b[2]:.4f}" for b in batches],
                                 hovertemplate="%{y:.2f} km, %{text}"), secondary_y=False)
    fig.add_hline(y=ref, line_dash="dash", annotation_text=f"reference {ref:g} km", secondary_y=False)
    fig.update_yaxes(title_text="grounding line (km)", secondary_y=False)
    fig.update_yaxes(title_text="rms change", type="log", secondary_y=True)
    fig.update_xaxes(title_text="model time (yr)")
    last = chunks[-1]
    fig.update_layout(title=f"{os.path.basename(log)}: grounding line {last[1]:.2f} km after {last[0]:g} yr "
                            f"({last[1] - (start if start is not None else ref):+.2f} km since the start)")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}: {len(chunks)} chunks, last grounding line {last[1]:.2f} km, rms change {last[2]:.3e}")


main()
