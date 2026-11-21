#!/usr/bin/env python3
"""The grounded surface for a temperate patch with its head at HEAD km, extrapolated from the converged sweep points.

The observation behind it (the refined sweep): the temperate part (head .. grounding line) keeps the same profile in
every steady state; only the cold (subtemperate) part upstream of the head changes shape. So:

    temperate, x >= HEAD:   the newest point's surface (absolute x): the common temperate curve;
    cold,      x <  HEAD:   a self-similar dome on it, Zs = Zs_T(HEAD) + a f(x / HEAD): f (f(0) = 1, f(1) = 0) the
                            newest point's shape, the height a = Zs(divide) - Zs(head) by a power law a ~ head^p
                            fitted over the last --fit points (default 8; the points' a and p, the shape's drift printed).

No jump at the head. Writes FILE.npz (x m, zs m, head_km, xgl_km, ...) for jobs/sia_start.py --surface, and FILE.html:
the points' surfaces, the start and the --compare states' surfaces.

    python3 plots/extrap_surface.py ROOT [ROOT2 ...] --head-km 894.4 [--fit 8] [--last N] [--compare S.npz ...]
                                    [--out FILE]

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
    args, fit = take(args, "--fit", conv=int, default=8)
    args, last = take(args, "--last", conv=int, default=0)
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
    # the cold dome: Zs(x) = Zs(head) + a(head) f(x / head), f(0) = 1, f(1) = 0. Zs(head) is the common temperate curve
    # (the newest point's surface at the new head); the height a by a power law a ~ head^p fitted (log-log) over the
    # last --fit points; the shape f the newest point's (its drift over the points printed)
    zh = cold[:, -1]
    amp = cold[:, 0] - zh
    f = (cold - zh[:, None]) / amp[:, None]
    nf = min(fit, len(ds))
    pw, lc = np.polyfit(np.log(heads[-nf:]), np.log(amp[-nf:]), 1)
    zt = float(np.interp(head_t, xs_n, zs_n))
    a_t = float(np.exp(lc) * head_t ** pw)
    pw_all = np.polyfit(np.log(heads), np.log(amp), 1)[0]
    zc = zt + a_t * f[-1]
    xc = S * head_t
    warm = xs_n > head_t
    x = np.r_[xc[:-1], head_t, xs_n[warm]]
    z = np.r_[zc[:-1], zt, zs_n[warm]]
    print(f"the dome height a = Zs(divide) - Zs(head): {amp[0]:.1f} m (head {heads[0]:.1f} km) .. {amp[-1]:.1f} m "
          f"(head {heads[-1]:.1f} km); a ~ head^p: p = {pw:.3f} over the last {nf} points ({pw_all:.3f} over all)")
    print(f"the dome shape f(x / head): max change over all points {np.abs(f[-1] - f[0]).max():.3f}, over the last "
          f"{nf} {np.abs(f[-1] - f[-nf]).max():.3f} (0 = self-similar)")
    print(f"at the head {head_t:g} km ({head_t - heads.max():+.1f} km beyond the last point): Zs(head) {zt:.1f} m (the "
          f"temperate curve), a {a_t:.1f} m -> Zs(divide) {zc[0]:.1f} m, Zs(x = {0.5 * head_t:.0f} km) "
          f"{np.interp(0.5, S, zc):.1f} m")
    out = out or os.path.join(os.path.dirname(os.path.normpath(args[0])), f"extrap_head{int(round(head_t)):04d}")
    np.savez(out + ".npz", x=x * 1e3, zs=z, head_km=head_t, xgl_km=xgl_n, deg=0, jump_m=0.0, power=pw,
             amp_m=a_t, heads_km=heads, d_km=np.array(ds))
    print(f"-> {out}.npz")
    import plotly.graph_objects as go
    fig = go.Figure()
    n = len(prof)
    for i, (d, h, xgl, xs, zs) in enumerate(prof):
        g = xs <= xgl + 1e-3
        fig.add_trace(go.Scatter(x=xs[g], y=zs[g], mode="lines", name=f"d {d:g} km (head {h:.0f})",
                                 line=dict(width=1, color=f"rgba(60,90,160,{0.15 + 0.8 * i / max(n - 1, 1):.2f})"),
                                 legendgroup="pts", showlegend=i in (0, n - 1)))
    fig.add_trace(go.Scatter(x=x, y=z, mode="lines", name=f"the start (dome height {a_t:.0f} m, p = {pw:.2f})",
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
