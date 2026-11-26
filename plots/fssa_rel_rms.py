#!/usr/bin/env python3
"""
Relative rms error of every packed land-strip run against the full-FSSA dt 10 run, at each output time: the rms of
|Zs - Zs_ref| divided by the rms change of the reference since its first frame (the same number as the last column of
plots/fssa_compare.py), as a plotly page. Full-FSSA runs dashed, implicit (imp*) runs solid; BDF 2 runs have square markers.
The reference's first frame has no change to divide by and is left out.

    python3 plots/fssa_rel_rms.py [FRAMES_DIR] [--out FILE.html] [--ref cmp_full_dt10_frames.npz]
    FRAMES_DIR: where the cmp_*_frames.npz are (plots/fssa_pack.sh on tethys, rsynced); default ~/Downloads
"""
import glob
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree


def load(p):
    z = np.load(p)
    return np.asarray(z["frame_times"], float), np.asarray(z["top_x"], float), np.asarray(z["top_y"], float), np.asarray(z["top_z"], float)


def rel_rms(ref, path):
    tr, xr, yr, zr = ref
    t, x, y, z = load(path)
    d, idx = cKDTree(np.column_stack([xr, yr])).query(np.column_stack([x, y]))
    if d.max() > 5 or len(np.unique(idx)) != len(idx):
        return None
    order = np.empty(len(idx), int)
    order[idx] = np.arange(len(idx))
    zz = z[:, order]
    out = []
    for k, tk in enumerate(t):
        j = int(np.argmin(np.abs(tr - tk)))
        if abs(tr[j] - tk) > 1.0:
            continue
        change = float(np.sqrt(np.mean((zr[j] - zr[0]) ** 2)))
        if change < 1e-9:
            continue
        out.append((tk, 100 * float(np.sqrt(np.mean((zz[k] - zr[j]) ** 2))) / change))
    return out


def main():
    a = sys.argv[1:]
    out = "fssa_rel_rms.html"
    refname = "cmp_full_dt10_frames.npz"
    if "--out" in a:
        i = a.index("--out"); out = a[i + 1]; del a[i:i + 2]
    if "--ref" in a:
        i = a.index("--ref"); refname = a[i + 1]; del a[i:i + 2]
    D = os.path.expanduser(a[0] if a else "~/Downloads")
    ref = load(os.path.join(D, refname))
    import plotly.graph_objects as go
    fig = go.Figure()
    dts = sorted({int(m.group(1)) for p in glob.glob(os.path.join(D, "cmp_*_frames.npz"))
                  for m in [re.search(r"dt(\d+)", os.path.basename(p))] if m})
    pal = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666", "#1f78b4", "#b2182b"]
    col = {d: pal[i % len(pal)] for i, d in enumerate(dts)}
    n = 0
    for p in sorted(glob.glob(os.path.join(D, "cmp_*_frames.npz"))):
        lab = os.path.basename(p)[4:-len("_frames.npz")]
        if os.path.basename(p) == refname or not (lab.startswith("full") or lab.startswith("imp")):
            continue
        r = rel_rms(ref, p)
        if not r:
            print(f"{lab}: no frames at the reference times (or other mesh), left out")
            continue
        m = re.search(r"dt(\d+)", lab)
        c = col.get(int(m.group(1)), "#444") if m else "#444"
        full = lab.startswith("full")
        fig.add_trace(go.Scatter(x=[q[0] for q in r], y=[q[1] for q in r], name=lab, mode="lines+markers",
                                 line=dict(color=c, dash="dash" if full else "solid", width=2),
                                 marker=dict(size=7, symbol="square" if "bdf2" in lab else "circle"),
                                 hovertemplate=lab + "<br>t %{x:.0f} yr<br>%{y:.2f} %<extra></extra>"))
        n += 1
        print(f"{lab}: {len(r)} points, last {r[-1][1]:.2f} % at t {r[-1][0]:g}")
    fig.update_layout(template="plotly_white", height=700, title="relative rms error of Zs against full FSSA dt 10 (dashed: full FSSA, solid: implicit, squares: BDF 2)",
                      xaxis_title="t (yr)", yaxis_title="rms(Zs - Zs_ref) / rms(Zs_ref(t) - Zs_ref(t0))  (%)", yaxis_type="log", hovermode="closest")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"{n} runs -> {out}")


if __name__ == "__main__":
    main()
