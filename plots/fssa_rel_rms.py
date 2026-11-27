#!/usr/bin/env python3
"""
Relative rms error of every packed land-strip run against the full-FSSA dt 10 run, at each output time: the rms of
|Zs - Zs_ref| divided by the rms change of the reference since the START of the runs (the 78,000 yr state: start_surface.npz from
plots/fssa_start_surface.py, so all 10 output frames have a number), as a plotly page. Full-FSSA runs dashed, implicit (imp*) runs
solid; BDF 2 runs have square markers. --norm first: the change since the reference's first frame instead (the same number as the
last column of plots/fssa_compare.py; its first frame has no change to divide by and is left out; also the fallback without
start_surface.npz).

    python3 plots/fssa_rel_rms.py [FRAMES_DIR] [--out FILE.html] [--ref cmp_full_dt10_frames.npz] [--norm start|first] [--t0 78000]
    FRAMES_DIR: where the cmp_*_frames.npz and start_surface.npz are (plots/fssa_pack.sh and plots/fssa_start_surface.py on tethys,
    rsynced); default ~/Downloads
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


def start_on(ref, start):
    """the start state's Zs on the reference's surface nodes (None: the node sets do not match)"""
    if start is None:
        return None
    sx, sy, sz = start
    d, idx = cKDTree(np.column_stack([sx, sy])).query(np.column_stack([ref[1], ref[2]]))
    if d.max() > 5 or len(np.unique(idx)) != len(idx):
        print(f"start surface: does not match the reference's nodes (max distance {d.max():g} m, {len(np.unique(idx))} of {len(idx)} distinct): --norm first")
        return None
    return sz[idx]


def rel_rms(ref, path, z0=None):
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
        change = float(np.sqrt(np.mean((zr[j] - (zr[0] if z0 is None else z0)) ** 2)))
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
    t0 = 78000.0                               # the start state (work/land_flowline_3d/state_078000.npz): the runs' time 0
    if "--t0" in a:
        i = a.index("--t0"); t0 = float(a[i + 1]); del a[i:i + 2]
    norm = "start"
    if "--norm" in a:
        i = a.index("--norm"); norm = a[i + 1]; del a[i:i + 2]
    if "--ref" in a:
        i = a.index("--ref"); refname = a[i + 1]; del a[i:i + 2]
    D = os.path.expanduser(a[0] if a else "~/Downloads")
    ref = load(os.path.join(D, refname))
    z0 = None
    sf = os.path.join(D, "start_surface.npz")
    if norm == "start":
        if os.path.exists(sf):
            q = np.load(sf)
            z0 = start_on(ref, (np.asarray(q["x"], float), np.asarray(q["y"], float), np.asarray(q["zs"], float)))
        else:
            print(f"{sf} not found (python3 plots/fssa_start_surface.py on tethys): --norm first")
    if z0 is not None:
        print(f"normalised by the reference's change since the start state: rms {np.sqrt(np.mean((ref[3][0] - z0) ** 2)):.3g} m at the first frame ({ref[0][0]:g} yr)")
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
        r = rel_rms(ref, p, z0)
        if not r:
            print(f"{lab}: no frames at the reference times (or other mesh), left out")
            continue
        m = re.search(r"dt(\d+)", lab)
        c = col.get(int(m.group(1)), "#444") if m else "#444"
        full = lab.startswith("full")
        fig.add_trace(go.Scatter(x=[q[0] - t0 for q in r], y=[q[1] for q in r], name=lab, mode="lines+markers",
                                 line=dict(color=c, dash="dash" if full else "solid", width=2),
                                 marker=dict(size=7, symbol="square" if "bdf2" in lab else "circle"),
                                 hovertemplate=lab + "<br>%{x:.0f} yr into the run<br>%{y:.2f} %<extra></extra>"))
        n += 1
        print(f"{lab}: {len(r)} points, last {r[-1][1]:.2f} % at {r[-1][0] - t0:g} yr into the run")
    fig.update_layout(template="plotly_white", height=700, title="relative rms error of Zs against full FSSA dt 10 (dashed: full FSSA, solid: implicit, squares: BDF 2)",
                      xaxis=dict(title="time since the start (yr)", range=[0, 10000]), yaxis_title="rms(Zs - Zs_ref) / rms(Zs_ref(t) - Zs_ref(" + ("start" if z0 is not None else "first frame") + "))  (%)", yaxis_type="log", hovermode="closest")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"{n} runs -> {out}")


if __name__ == "__main__":
    main()
