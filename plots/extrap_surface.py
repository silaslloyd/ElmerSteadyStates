#!/usr/bin/env python3
"""The grounded surface for a temperate patch with its head at HEAD km, extrapolated from the converged sweep points.

The observation behind it (the refined sweep): the temperate part (head .. grounding line) keeps the same profile in
every steady state; only the cold (subtemperate) part upstream of the head changes shape. So:

    temperate, x >= HEAD:   the newest point's surface (absolute x);
    cold,      x <  HEAD:   every point's surface in s = x / head (0 = divide, 1 = its head), each s extrapolated
                            in the head position (a polynomial of degree DEG over the points) to HEAD.

The two parts are joined at HEAD: the extrapolated cold surface's jump to the temperate one there is printed and
taken out over the last --blend-km of the cold part (linearly). Writes FILE.npz (x m, zs m, head_km, xgl_km, deg,
jump_m) for jobs/sia_start.py --surface, and FILE.html: the points' surfaces, the extrapolations of degree 1, 2, 3
(the chosen one solid) and the --compare states' surfaces.

    python3 plots/extrap_surface.py WORK_SWEEP_ROOT [ROOT2 ...] --head-km 894.4 [--deg 2] [--last N] [--blend-km 20]
                                    [--compare S.npz ...] [--out FILE]

ROOT: a refined sweep's folder (seg_*/sweep.csv, points/state_*.npz), e.g. work/flowline_rect_coarse/sweep_refined.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sweep_refined import converged  # noqa: E402
from sia_start import column_profile  # noqa: E402
from ess.spectrum import gl_position  # noqa: E402
from ess.statefile import load_state  # noqa: E402


def take(args, name, n=1, conv=float, default=None, many=False):
    if name not in args:
        return args, default
    k = args.index(name)
    if many:
        j = k + 1
        while j < len(args) and not args[j].startswith("--"):
            j += 1
        return args[:k] + args[j:], [conv(v) for v in args[k + 1:j]]
    return args[:k] + args[k + 2:], conv(args[k + 1])


def main():
    args = sys.argv[1:]
    args, head_t = take(args, "--head-km")
    args, deg = take(args, "--deg", conv=int, default=2)
    args, last = take(args, "--last", conv=int, default=0)
    args, blend = take(args, "--blend-km", default=20.0)
    args, out = take(args, "--out", conv=str)
    args, comp = take(args, "--compare", conv=str, default=[], many=True)
    if not args or head_t is None:
        raise SystemExit(__doc__)
    pts = {}
    for r in args:
        for f in sorted(glob.glob(os.path.join(r, "seg_*"))) or [r]:
            for d, p in converged(f):
                if os.path.exists(p):
                    pts[round(d, 1)] = p                # a point in two segments (the segment's start): once
    ds = sorted(pts, reverse=True)                      # the longest patch first, the newest last
    if last:
        ds = ds[-last:]
    if len(ds) < 2:
        raise SystemExit(f"{len(ds)} converged points found")
    S = np.linspace(0.0, 1.0, 801)
    heads, xgls, cold, prof = [], [], [], []
    for d in ds:
        s = load_state(pts[d])
        xgl = gl_position(s.frozen, s.diag) / 1e3
        h = xgl - d
        xs, zs = column_profile(s, "zs")
        xs = xs / 1e3
        heads.append(h)
        xgls.append(xgl)
        cold.append(np.interp(S * h, xs, zs))
        prof.append((d, h, xgl, xs, zs))
        print(f"d {d:7.1f} km  head {h:7.2f} km  grounding line {xgl:8.2f} km  Zs(divide) {zs[0]:7.1f} m  "
              f"Zs(head) {np.interp(h, xs, zs):7.1f} m")
    heads, cold = np.array(heads), np.array(cold)
    _, _, xgl_n, xs_n, zs_n = prof[-1]
    hc = heads.mean()

    def extrap(k):
        k = min(k, len(ds) - 1)
        c = np.polyfit(heads - hc, cold, k)                  # every s at once (columns of cold)
        return sum(c[i] * (head_t - hc) ** (k - i) for i in range(k + 1))

    zt = float(np.interp(head_t, xs_n, zs_n))           # the temperate surface at the new head (newest point)
    res = {}
    for k in (1, 2, 3):
        zc = extrap(k)
        res[k] = (zc, float(zt - zc[-1]))
    zc, jump = res[deg]
    xc = S * head_t
    w = np.clip(1.0 - (head_t - xc) / max(blend, 1e-9), 0.0, 1.0)
    zc = zc + jump * w
    warm = xs_n > head_t
    x = np.r_[xc[:-1], head_t, xs_n[warm]]
    z = np.r_[zc[:-1], zt, zs_n[warm]]
    print(f"extrapolated from {len(ds)} points (heads {heads.min():.1f} .. {heads.max():.1f} km) to the head at "
          f"{head_t:g} km ({head_t - heads.max():+.1f} km beyond the last)")
    for k in (1, 2, 3):
        zk, jk = res[k]
        print(f"  degree {k}: Zs(divide) {zk[0]:7.1f} m, Zs(x = {0.5 * head_t:.0f} km) {np.interp(0.5, S, zk):7.1f} m, "
              f"jump to the temperate surface at the head {jk:+.1f} m" + ("   <- used" if k == deg else ""))
    out = out or os.path.join(os.path.dirname(os.path.normpath(args[0])), f"extrap_head{int(round(head_t)):04d}")
    np.savez(out + ".npz", x=x * 1e3, zs=z, head_km=head_t, xgl_km=xgl_n, deg=deg, jump_m=jump,
             heads_km=heads, d_km=np.array(ds))
    print(f"-> {out}.npz")
    import plotly.graph_objects as go
    fig = go.Figure()
    n = len(prof)
    for i, (d, h, xgl, xs, zs) in enumerate(prof):
        g = xs <= xgl + 1e-3
        fig.add_trace(go.Scatter(x=xs[g], y=zs[g], mode="lines", name=f"d {d:g} km (head {h:.0f})",
                                 line=dict(width=1, color=f"rgba(60,90,160,{0.15 + 0.8 * i / max(n - 1, 1):.2f})"),
                                 legendgroup="pts", showlegend=i in (0, n - 1)))
    for k in (1, 2, 3):
        zk, jk = res[k]
        fig.add_trace(go.Scatter(x=xc, y=zk, mode="lines", name=f"cold part, degree {k} (jump {jk:+.0f} m)",
                                 line=dict(width=2, dash=None if k == deg else "dot"),
                                 visible=True if k == deg else "legendonly"))
    fig.add_trace(go.Scatter(x=x, y=z, mode="lines", name=f"the start (degree {deg}, joined)",
                             line=dict(width=3, color="black")))
    for p in comp:
        if not os.path.exists(p):
            print(f"--compare {p}: not found, skipped")
            continue
        s = load_state(p)
        xs, zs = column_profile(s, "zs")
        fig.add_trace(go.Scatter(x=xs / 1e3, y=zs, mode="lines", name=os.path.basename(p),
                                 line=dict(width=2, dash="dash"), visible="legendonly"))
    xb, zb = column_profile(load_state(pts[ds[-1]]), "zb")
    fig.add_trace(go.Scatter(x=xb / 1e3, y=zb, mode="lines", name="Zb (newest point)",
                             line=dict(color="#8e44ad", width=1)))
    fig.add_vline(x=head_t, line=dict(color="red", dash="dash", width=1))
    fig.update_layout(template="plotly_white", xaxis_title="x (km)", yaxis_title="z (m)",
                      title=f"surface extrapolated to the patch head at {head_t:g} km (cold part from {len(ds)} points)")
    fig.write_html(out + ".html", include_plotlyjs="cdn")
    print(f"-> {out}.html")


if __name__ == "__main__":
    main()
