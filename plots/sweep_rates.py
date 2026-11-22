#!/usr/bin/env python3
"""How fast the converged sweep states change with the patch length d: between successive points (any meshes: the
profiles compared on a common x grid), per km of d, the surface Zs (rms and max |dZs|/dd, m/km) and the grounded basal
temperature (rms and max |dTb|/dd, K/km, over the bed cold in both), plus each point's Newton iterations. Rates growing
without bound as d decreases mark a fold of the branch ahead (natural steps in d cannot pass it); level rates say the
states change smoothly, and Newton's trouble is elsewhere.

    python3 plots/sweep_rates.py ROOT [ROOT2 ...] [--out FILE.html]

ROOT: a sweep folder (seg_*/sweep.csv + points/) or a single sweep's folder (sweep.csv + points/).
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sia_start import column_profile  # noqa: E402
from ess.spectrum import _bed, basal_temperature  # noqa: E402
from ess.statefile import load_state  # noqa: E402


def points(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "seg_*", "sweep.csv"))) or [os.path.join(root, "sweep.csv")]:
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            if str(r.get("converged")) != "1":
                continue
            d = round(float(r["d_km"]), 1)
            p = os.path.join(os.path.dirname(f), "points", f"state_{int(round(d * 10)):05d}.npz")
            if os.path.exists(p):
                out[d] = (p, int(float(r.get("newton_its") or 0)))
    return out


def main():
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        k = args.index("--out")
        out = args[k + 1]
        args = args[:k] + args[k + 2:]
    if not args:
        raise SystemExit(__doc__)
    pts = {}
    for r in args:
        pts.update(points(r))
    ds = sorted(pts, reverse=True)
    xg = np.arange(0.0, 1100.0, 1.0)
    prev = None
    rows = []
    print(f"{'d km':>7} {'its':>4} {'step':>6} {'Zs rms':>8} {'Zs max':>8} {'at km':>7} {'Tb rms':>8} {'Tb max':>8} {'at km':>7}"
          "   (per km of d: m/km, K/km)")
    for d in ds:
        p, its = pts[d]
        s = load_state(p)
        xs, zs = column_profile(s, "zs")
        g, xpos, gr = _bed(s.frozen, s.diag)
        T = basal_temperature(s.layout, s.x, g) - 273.15
        o = np.argsort(xpos[gr])
        xb, kb = np.unique(np.round(xpos[gr][o] / 1e3, 3), return_index=True)
        Tb = T[gr][o][kb]
        head = float(s.meta.get("x_gl", np.nan)) / 1e3 - d if "x_gl" in s.meta else np.nan
        cur = dict(d=d, Z=np.interp(xg, xs / 1e3, zs), T=np.interp(xg, xb, Tb, right=np.nan),
                   cold=np.interp(xg, xb, Tb, right=np.nan) < -1e-3, gl=xb.max(), head=head)
        if prev is not None:
            dd = abs(prev["d"] - d)
            dz = (cur["Z"] - prev["Z"]) / dd
            m = (xg < min(cur["gl"], prev["gl"]))
            dz = np.where(m, dz, np.nan)
            mt = m & cur["cold"] & prev["cold"]
            dt = np.where(mt, (cur["T"] - prev["T"]) / dd, np.nan)
            kz, kt = int(np.nanargmax(np.abs(dz))), int(np.nanargmax(np.abs(dt))) if mt.any() else 0
            row = dict(d=d, its=its, step=dd, zr=float(np.sqrt(np.nanmean(dz ** 2))), zm=float(np.abs(dz[kz])),
                       zx=float(xg[kz]), tr=float(np.sqrt(np.nanmean(dt ** 2))) if mt.any() else np.nan,
                       tm=float(np.abs(dt[kt])) if mt.any() else np.nan, tx=float(xg[kt]))
            rows.append(row)
            print(f"{d:7.1f} {its:4d} {dd:6.2f} {row['zr']:8.3f} {row['zm']:8.3f} {row['zx']:7.0f} {row['tr']:8.4f} "
                  f"{row['tm']:8.4f} {row['tx']:7.0f}")
        prev = cur
    if out and rows:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("surface change per km of d (m/km)", "basal temperature change per km of d (K/km)",
                                            "Newton iterations"))
        d = [r["d"] for r in rows]
        fig.add_trace(go.Scatter(x=d, y=[r["zr"] for r in rows], name="Zs rms", mode="lines+markers"), 1, 1)
        fig.add_trace(go.Scatter(x=d, y=[r["zm"] for r in rows], name="Zs max", mode="lines+markers"), 1, 1)
        fig.add_trace(go.Scatter(x=d, y=[r["tr"] for r in rows], name="Tb rms", mode="lines+markers"), 2, 1)
        fig.add_trace(go.Scatter(x=d, y=[r["tm"] for r in rows], name="Tb max", mode="lines+markers"), 2, 1)
        fig.add_trace(go.Scatter(x=d, y=[r["its"] for r in rows], name="Newton its", mode="lines+markers"), 3, 1)
        fig.update_xaxes(autorange="reversed", title_text="patch length d (km)", row=3, col=1)
        fig.update_layout(template="plotly_white", height=900, title="how fast the sweep's states change with d")
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
