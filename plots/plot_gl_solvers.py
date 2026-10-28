#!/usr/bin/env python3
"""
The moving-grounding-line solver test (configs/test_gl_solvers.toml, jobs/gl_solvers.sh) in 2D: the flowline
profile of every solver run overlaid, with a slider through the chunks (spin-up, retreat, advance). Panels:
  1. geometry along the flow: surface Zs and ice base Zb of each run, the bed (grey), the grounding lines
  2. surface speed of each run
  3. the grounding line (the upstream edge of the shelf) of each run against chunk, the current chunk marked
All solvers solve the same problem, so the runs should lie on top of each other; a run that drifts apart in
panel 3 did not converge.

    python plots/plot_gl_solvers.py                                   # work/test_gl_solvers/{direct,iterative,block,amg}
    python plots/plot_gl_solvers.py work/test_gl_solvers --x 650 820  # x window (km)
    python plots/plot_gl_solvers.py WORK_DIR direct block             # given runs only
Writes <work dir>/gl_solvers.html and opens it. Needs numpy and plotly.
"""
import glob
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ess.statefile import find_name, load_state  # noqa: E402

PHASES = ("spinup", "retreat", "advance")
SOLVERS = ("direct", "iterative", "block", "amg")
COLOURS = {"direct": "#1f77b4", "iterative": "#d62728", "block": "#2ca02c", "amg": "#9467bd"}
X_WINDOW = None          # (xmin, xmax) km; None: 80 km behind the most inland grounding line to the front
SHOW = True


def _rec(s, name):
    allf = dict(s.diag)
    allf.update(s.frozen)
    blocks = s.layout.split(s.x)
    for b in s.layout.blocks:
        allf[b.name] = (b.gids, blocks[b.name])
    try:
        return allf[find_name(allf, name)]
    except KeyError:
        return None


def profile(path):
    """the y = 0 row of the flowline: x (km) and Zs, speed at the surface; Zb, bed, mask at the base"""
    s = load_state(path)
    g, X = _rec(s, "coordinate 1")
    Y = dict(zip(g.tolist(), _rec(s, "coordinate 2")[1].tolist()))
    Xd = dict(zip(g.tolist(), X.tolist()))
    u = [_rec(s, f"Velocity {k}") for k in (1, 2, 3)]
    sp = {}
    if u[0] is not None:
        d = [dict(zip(r[0].tolist(), r[1].tolist())) for r in u if r is not None]
        sp = {i: float(np.sqrt(sum(dk.get(i, 0.0) ** 2 for dk in d))) for i in d[0]}

    def row(name, extra=()):
        r = _rec(s, name)
        if r is None:
            return None
        gi = [i for i in r[0].tolist() if abs(Y[i]) < 1.0]
        v = dict(zip(r[0].tolist(), r[1].tolist()))
        gi.sort(key=lambda i: Xd[i])
        out = dict(x=np.array([Xd[i] for i in gi]) / 1e3, v=np.array([v[i] for i in gi]))
        for nm in extra:
            e = _rec(s, nm)
            ed = dict(zip(e[0].tolist(), e[1].tolist())) if e is not None else {}
            out[nm] = np.array([ed.get(i, np.nan) for i in gi])
        out["speed"] = np.array([sp.get(i, np.nan) for i in gi])
        return out
    top = row("Zs")
    base = row("Zb", ("bedrock", "GroundedMask"))
    gl = np.nan
    if base is not None and np.isfinite(base["GroundedMask"]).any():
        # the upstream edge of the shelf: the last grounded node before the first floating one (grounded patches
        # under the shelf do not count)
        fl = np.where(base["GroundedMask"] < -0.5)[0]
        gl = float(base["x"][fl[0] - 1]) if len(fl) and fl[0] > 0 else (float(base["x"][-1]) if not len(fl) else np.nan)
        base["patches"] = int(np.sum(base["GroundedMask"][fl[0]:] > -0.5)) if len(fl) else 0
    return dict(top=top, base=base, gl=gl, t=float(s.meta.get("time", np.nan)))


def main():
    import plotly.graph_objects as go
    global X_WINDOW
    args = sys.argv[1:]
    if "--x" in args:
        k = args.index("--x")
        X_WINDOW = (float(args[k + 1]), float(args[k + 2]))
        args = args[:k] + args[k + 3:]
    work = args[0].rstrip("/") if args else "work/test_gl_solvers"
    runs = args[1:] or [r for r in SOLVERS if os.path.isdir(os.path.join(work, r))]
    if not runs:
        raise SystemExit(f"no runs ({', '.join(SOLVERS)}) in {work}")
    files = {}
    for r in runs:
        for ph in PHASES:
            for f in sorted(glob.glob(os.path.join(work, r, f"{ph}_chunks", "points", "state_*.npz"))):
                files[(r, ph, int(os.path.basename(f)[6:10]))] = f
    keys = sorted({(ph, c) for _, ph, c in files}, key=lambda k: (PHASES.index(k[0]), k[1]))
    if not keys:
        raise SystemExit(f"no chunk states in {work}/<run>/<phase>_chunks/points")
    print(f"{len(runs)} runs ({', '.join(runs)}), {len(keys)} chunks")
    P = {}
    for (r, ph, c), f in files.items():
        P[(r, ph, c)] = profile(f)
    # grounding-line history (panel 3): chunk index along the slider
    hist = {r: [P[(r, *k)]["gl"] if (r, *k) in P else np.nan for k in keys] for r in runs}
    for r in runs:
        g = np.array(hist[r], float)
        if np.isfinite(g).any():
            pt = [P[(r, *k)]["base"].get("patches", 0) for k in keys if (r, *k) in P and P[(r, *k)]["base"] is not None]
            print(f"  {r}: grounding line {np.nanmin(g):.2f} .. {np.nanmax(g):.2f} km; grounded nodes under the shelf "
                  f"(downstream of the grounding line): up to {max(pt) if pt else 0}")
    if X_WINDOW:
        x0, x1 = X_WINDOW
    else:
        gls = [v for h in hist.values() for v in h if np.isfinite(v)]
        xmax = max(float(p["top"]["x"].max()) for p in P.values() if p["top"] is not None)
        x0, x1 = (min(gls) - 80.0 if gls else 0.0), xmax
    zs = [v for p in P.values() for d in (p["top"], p["base"]) if d is not None
          for v in (np.nanmin(d["v"]), np.nanmax(d["v"]))]
    zs += [np.nanmin(p["base"]["bedrock"]) for p in P.values() if p["base"] is not None]
    smax = max(float(np.nanmax(p["top"]["speed"])) for p in P.values() if p["top"] is not None)
    gall = [v for h in hist.values() for v in h if np.isfinite(v)]
    labels = [f"{ph} {c}" for ph, c in keys]

    def crop(d):
        m = (d["x"] >= x0) & (d["x"] <= x1)
        return {k: (v[m] if isinstance(v, np.ndarray) and v.shape == d["x"].shape else v) for k, v in d.items()}

    def traces(n):
        tr = []
        k = keys[n]
        bed_done = False
        for r in runs:
            col = COLOURS.get(r, None)
            p = P.get((r, *k))
            top = crop(p["top"]) if p and p["top"] is not None else None
            base = crop(p["base"]) if p and p["base"] is not None else None
            if not bed_done:
                b = base if base is not None else dict(x=[], bedrock=[])
                tr.append(go.Scatter(x=b["x"], y=b["bedrock"], mode="lines", line=dict(color="#888888", width=2),
                                     name="bed", xaxis="x", yaxis="y",
                                     legendgroup="bed", showlegend=n == 0))
                bed_done = True
            tr.append(go.Scatter(x=top["x"] if top else [], y=top["v"] if top else [], mode="lines",
                                 line=dict(color=col, width=2), name=f"{r}", legendgroup=r, showlegend=True,
                                 xaxis="x", yaxis="y", hovertemplate=f"{r} Zs %{{y:.1f}} m at %{{x:.2f}} km<extra></extra>"))
            tr.append(go.Scatter(x=base["x"] if base else [], y=base["v"] if base else [], mode="lines",
                                 line=dict(color=col, width=2, dash="dot"), name=f"{r} base", legendgroup=r,
                                 showlegend=False, xaxis="x", yaxis="y",
                                 hovertemplate=f"{r} Zb %{{y:.1f}} m at %{{x:.2f}} km<extra></extra>"))
            gl = p["gl"] if p else np.nan
            tr.append(go.Scatter(x=[gl, gl] if np.isfinite(gl) else [], y=[min(zs), max(zs)] if np.isfinite(gl) else [],
                                 mode="lines", line=dict(color=col, width=1, dash="dash"), legendgroup=r,
                                 showlegend=False, xaxis="x", yaxis="y", name=f"{r} grounding line",
                                 hovertemplate=f"{r} grounding line %{{x:.2f}} km<extra></extra>"))
            tr.append(go.Scatter(x=top["x"] if top else [], y=top["speed"] if top else [], mode="lines",
                                 line=dict(color=col, width=2), legendgroup=r, showlegend=False, xaxis="x2",
                                 yaxis="y2", name=f"{r} speed",
                                 hovertemplate=f"{r} %{{y:.1f}} m/yr at %{{x:.2f}} km<extra></extra>"))
            tr.append(go.Scatter(x=list(range(len(keys))), y=hist[r], mode="lines", line=dict(color=col, width=2),
                                 legendgroup=r, showlegend=False, xaxis="x3", yaxis="y3", name=f"{r} GL history",
                                 customdata=labels,
                                 hovertemplate=f"{r} %{{customdata}}: %{{y:.2f}} km<extra></extra>"))
            tr.append(go.Scatter(x=[n], y=[hist[r][n]], mode="markers", marker=dict(color=col, size=10),
                                 legendgroup=r, showlegend=False, xaxis="x3", yaxis="y3", hoverinfo="skip"))
        return tr

    frames = [go.Frame(data=traces(n), name=str(n)) for n in range(len(keys))]
    fig = go.Figure(data=traces(len(keys) - 1), frames=frames)
    steps = [dict(method="animate", label=lab, args=[[str(n)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                                     transition=dict(duration=0))])
             for n, lab in enumerate(labels)]
    # phase boundaries in panel 3
    shapes, ann = [], []
    for ph in PHASES:
        idx = [n for n, k in enumerate(keys) if k[0] == ph]
        if idx:
            ann.append(dict(x=(idx[0] + idx[-1]) / 2, y=1.0, xref="x3", yref="y3 domain", text=ph, showarrow=False,
                            yanchor="bottom"))
            if idx[0] > 0:
                shapes.append(dict(type="line", x0=idx[0] - 0.5, x1=idx[0] - 0.5, xref="x3", y0=0, y1=1,
                                   yref="y3 domain", line=dict(color="#999999", dash="dot")))
    fig.update_layout(
        title=f"{work}: {', '.join(runs)} (solid surface, dotted base, dashed grounding line)",
        xaxis=dict(domain=[0, 1], anchor="y", range=[x0, x1], title="x (km)"),
        yaxis=dict(domain=[0.52, 1.0], title="z (m)", range=[min(zs) - 50, max(zs) + 50]),
        xaxis2=dict(domain=[0, 1], anchor="y2", range=[x0, x1], matches="x", title="x (km)"),
        yaxis2=dict(domain=[0.30, 0.46], title="surface speed (m/yr)", range=[0, smax * 1.05]),
        xaxis3=dict(domain=[0, 1], anchor="y3", title="chunk (100 yr each)", range=[-0.5, len(keys) - 0.5]),
        yaxis3=dict(domain=[0.0, 0.22], title="grounding line (km)",
                    range=[min(gall) - 2, max(gall) + 2] if gall else None),
        shapes=shapes, annotations=ann,
        sliders=[dict(steps=steps, active=len(keys) - 1, currentvalue=dict(prefix="chunk: "), pad=dict(t=40))],
        template="plotly_white", autosize=True, margin=dict(l=60, r=20, t=60, b=20), legend=dict(x=1.0, y=1.0,
                                                                                               xanchor="right"))
    out = os.path.join(work, "gl_solvers.html")
    fig.write_html(out, include_plotlyjs="cdn", config=dict(responsive=True), default_width="100%",
                   default_height="95vh")
    print(f"wrote {out}")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
