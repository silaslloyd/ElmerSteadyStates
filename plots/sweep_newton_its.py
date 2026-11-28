#!/usr/bin/env python3
"""
Plotly page: Newton iterations per sweep point against the patch length d (one line per sweep), from the sweeps' sweep.csv files.
Hover: d, iterations, rms(F) at the end, the step taken, whether it converged. The steps-halving limit (8) is drawn; points that did
not converge are crosses. d runs from large to small left to right, the order of the sweep.

  python3 plots/sweep_newton_its.py --out page.html LABEL CSV [CSV ...] [-- LABEL CSV [CSV ...]] ...
Needs plotly.
"""
import csv
import sys


def main():
    a = sys.argv[1:]
    out = "sweep_newton_its.html"
    if "--out" in a:
        i = a.index("--out"); out = a[i + 1]; del a[i:i + 2]
    groups, cur = [], []
    for t in a + ["--"]:
        if t == "--":
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append(t)
    import plotly.graph_objects as go
    fig = go.Figure()
    for g in groups:
        label, files = g[0], g[1:]
        rows = []
        for f in files:
            try:
                rows += [r for r in csv.DictReader(open(f)) if r.get("newton_its", "") != ""]
            except OSError as e:
                print(f"skip {f}: {e}")
        if not rows:
            print(f"{label}: no rows")
            continue
        rows.sort(key=lambda r: -float(r["d_km"]))
        d = [float(r["d_km"]) for r in rows]
        n = [int(float(r["newton_its"])) for r in rows]
        ok = [str(r.get("converged")) == "1" for r in rows]
        hov = [f"d = {r['d_km']} km<br>Newton its {r['newton_its']}<br>rms(F) {float(r['rms_F']):.2e}<br>step {r.get('step_km', '')} km"
               f"<br>{'converged' if str(r.get('converged')) == '1' else 'NOT converged'}" for r in rows]
        fig.add_trace(go.Scatter(x=d, y=n, mode="lines+markers", name=label, text=hov, hoverinfo="text",
                                 marker=dict(size=8, symbol=["circle" if o else "x" for o in ok]), line=dict(width=1.5)))
        print(f"{label}: {len(rows)} points, d {d[0]:g} .. {d[-1]:g} km, iterations {min(n)}..{max(n)}, total {sum(n)}")
    fig.add_hline(y=8, line=dict(dash="dot", width=1), annotation_text="limit 8: next steps halved", annotation_position="top right")
    fig.update_layout(title="Newton iterations per sweep point", xaxis=dict(title="patch length d (km)", autorange="reversed"),
                      yaxis=dict(title="Newton iterations", rangemode="tozero", dtick=1), height=600, hovermode="closest")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
