#!/usr/bin/env python3
"""The grounded surface for a temperate patch with its head at HEAD km, extrapolated from the converged sweep points.

The observation behind it (the refined sweep): the temperate part (head .. grounding line) keeps the same profile in
every steady state; only the cold (subtemperate) part upstream of the head changes shape. So:

    temperate, x >= HEAD:   the newest point's surface (absolute x): the common temperate curve;
    cold, every x cold in >= --min-points states (default 6; cold: x <= head - --margin-km 5): its own history Zs(x)
          over the heads extrapolated: the gains per step geometric (log|gain| linear over the last --fit, default 10,
          states), summed on to HEAD (as the divide's: its gains shrink by ~0.95 per 12.5 km step). Up to the first x
          without enough cold states or with gains of mixed sign / not shrinking: x_e;
    bridge, x_e .. HEAD (never cold yet: no history): a cubic from the extrapolated end (value, slope) to the temperate
          curve at HEAD (value; the slope of the newest point's cold surface just above its head).

Writes FILE.npz (x m, zs m, head_km, xgl_km, x_extrap_km) for jobs/sia_start.py --surface, and FILE.html: the points'
surfaces, the start (the bridge in orange) and the --compare states' surfaces.

    python3 plots/extrap_surface.py ROOT [ROOT2 ...] --head-km 894.4 [--fit 10] [--min-points 6] [--margin-km 5]
                                    [--last N] [--compare S.npz ...] [--out FILE]

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
from ess.spectrum import _bed, basal_temperature, gl_position  # noqa: E402
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


def extrap_hist(Z, fit, minp, n):
    """every column of Z (rows: the states in order of the head, NaN where not cold): its last <= fit values' gains
    geometric (log|gain| linear), summed on n steps; NaN from the first column without minp values or with gains of
    mixed sign / not shrinking"""
    ze, ratio = np.full(Z.shape[1], np.nan), np.full(Z.shape[1], np.nan)
    for m in range(Z.shape[1]):
        v = Z[:, m]
        v = v[np.isfinite(v)][-fit:]
        if len(v) < minp:
            break
        g = np.diff(v)
        if not (np.all(g > 0) or np.all(g < 0)):
            break
        kk = np.arange(len(g))
        sl, ic = np.polyfit(kk, np.log(np.abs(g)), 1)
        r = float(np.exp(sl))
        if r >= 1.0:
            break
        g_last = np.sign(g[-1]) * np.exp(ic + sl * kk[-1])
        ze[m] = v[-1] + g_last * r * (1 - r ** n) / (1 - r)
        ratio[m] = r
    return ze, ratio


def main():
    args = sys.argv[1:]
    args, head_t = take(args, "--head-km")
    args, fit = take(args, "--fit", conv=int, default=10)
    args, minp = take(args, "--min-points", conv=int, default=6)
    args, margin = take(args, "--margin-km", default=5.0)
    args, tbfit = take(args, "--tb-fit-km", default=150.0)
    tbhist = "--tb-history" in args
    args = [v for v in args if v != "--tb-history"]
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
    heads, xgls, prof, tbp = [], [], [], []
    for d in ds:
        s = load_state(pts[d])
        xgl = gl_position(s.frozen, s.diag) / 1e3
        h = xgl - d
        xs, zs = column_profile(s, "zs")
        xs = xs / 1e3
        heads.append(h)
        xgls.append(xgl)
        prof.append((d, h, xgl, xs, zs))
        g, xpos, gr = _bed(s.frozen, s.diag)
        Tb = basal_temperature(s.layout, s.x, g) - 273.15
        o = np.argsort(xpos[gr])
        xu, ku = np.unique(np.round(xpos[gr][o] / 1e3, 3), return_index=True)
        tbp.append((xu, Tb[gr][o][ku]))
        print(f"d {d:7.1f} km  head {h:7.2f} km  grounding line {xgl:8.2f} km  Zs(divide) {zs[0]:7.1f} m  "
              f"Zs(head) {np.interp(h, xs, zs):7.1f} m")
    heads = np.array(heads)
    _, _, xgl_n, xs_n, zs_n = prof[-1]
    # 1. every x cold in >= --min-points states: its own history Zs(x) over the heads (the states with head > x +
    #    --margin-km), the gains per step fitted as geometric (log|gain| linear in the head over the last --fit),
    #    summed on to the new head. Stops at the first x without --min-points or with gains of mixed sign / growing.
    dh = float(np.median(np.diff(heads)))
    xg = np.arange(0.0, head_t, 2.0)
    Z = np.array([np.where(xg <= h - margin, np.interp(xg, xs, zs), np.nan) for (_, h, _, xs, zs) in prof])
    ze, ratio = extrap_hist(Z, fit, minp, (head_t - heads[-1]) / dh)
    ok = np.isfinite(ze)
    if not ok.any():
        raise SystemExit("no x with a usable history (lower --min-points?)")
    me = int(np.flatnonzero(ok)[-1])
    x_e, z_e = xg[me], ze[me]
    s_e = (ze[me] - ze[me - 3]) / (xg[me] - xg[me - 3]) if me >= 3 else 0.0
    # 2. the bridge x_e .. head: a cubic (Hermite) from the extrapolated end (its value, slope) to the temperate curve at
    #    the new head (its value; the slope of the newest point's cold surface just above its head)
    zt = float(np.interp(head_t, xs_n, zs_n))
    hn = heads[-1]
    s_t = (np.interp(hn - 1.0, xs_n, zs_n) - np.interp(hn - 6.0, xs_n, zs_n)) / 5.0
    L = head_t - x_e
    xb = np.linspace(x_e, head_t, max(int(L / 2.0), 2) + 1)
    u = (xb - x_e) / L
    zb_ = ((2 * u ** 3 - 3 * u ** 2 + 1) * z_e + (u ** 3 - 2 * u ** 2 + u) * L * s_e
           + (-2 * u ** 3 + 3 * u ** 2) * zt + (u ** 3 - u ** 2) * L * s_t)
    warm = xs_n > head_t
    x = np.r_[xg[:me + 1], xb[1:], xs_n[warm]]
    z = np.r_[ze[:me + 1], zb_[1:], zs_n[warm]]
    print(f"extrapolated pointwise: x = 0 .. {x_e:.0f} km (each from its last <= {fit} cold states, >= {minp}); "
          f"gain ratios per {dh:g} km step {np.nanmin(ratio):.3f} .. {np.nanmax(ratio):.3f}; to the head at {head_t:g} km "
          f"({head_t - hn:+.1f} km beyond the last point)")
    print(f"the bridge {x_e:.0f} .. {head_t:g} km: from {z_e:.1f} m (slope {s_e * 1e3 / 1e3:.3f} m/km) to the temperate "
          f"curve {zt:.1f} m (slope {s_t:.3f} m/km, the newest point's cold surface above its head)")
    print("Zs: " + "  ".join(f"{xx:.0f} km {np.interp(xx, x, z):.0f} m" for xx in (0, 100, 200, 300, 400, 500, 600, 700, 800)))
    # the basal temperature (T - 273.15 K, grounded): the same pointwise extrapolation where it has a usable history;
    # melting from the head on
    xn_t, Tn = tbp[-1]
    TZ = np.array([np.where(xg <= h - margin, np.interp(xg, xu, Tu), np.nan) for (_, h, _, _, _), (xu, Tu) in zip(prof, tbp)])
    te, tr = extrap_hist(TZ, fit, minp, (head_t - heads[-1]) / dh)
    if not tbhist:                  # default: the newest point's near the divide too (its history is slow, r -> 1: a kink)
        te[:] = np.nan
    okt = np.isfinite(te)
    mt = int(np.flatnonzero(okt)[-1]) if okt.any() else -1
    xtb = np.r_[xg, head_t, xs_n[warm]]
    # beyond: the states' cold beds lie on ONE curve in absolute x (they differ only near the divide); the newest
    # point's up to its head - margin, then that curve continued to the new head: T = -A exp(-x / L) fitted over its
    # last --tb-fit-km (150) cold km
    x_te = xg[mt] if mt >= 0 else 0.0
    x_c = hn - margin
    fw = (xn_t >= x_c - tbfit) & (xn_t <= x_c) & (Tn < -1e-6)
    sl, ic = np.polyfit(xn_t[fw], np.log(-Tn[fw]), 1)
    Lc = -1.0 / sl
    env = np.where(xtb <= x_c, np.interp(xtb, xn_t, Tn), -np.exp(ic + sl * xtb))
    tb = np.where(xtb <= x_te, np.interp(xtb, xg[:mt + 1], te[:mt + 1]) if mt >= 0 else env, env)
    tb = np.where(xtb >= head_t, 0.0, np.minimum(tb, 0.0))
    print(f"basal temperature: extrapolated pointwise to {x_te:.0f} km" + (f" (gain ratios {np.nanmin(tr):.3f} .. "
          f"{np.nanmax(tr):.3f})" if mt >= 0 else " (no usable history)") + f"; to {x_c:.0f} km the newest point's (the common curve), "
          f"beyond it continued as -A exp(-x / L), L = {Lc:.0f} km (fitted over {x_c - tbfit:.0f} .. {x_c:.0f} km): "
          f"{-np.exp(ic + sl * head_t):+.3f} K at the new head")
    print("T_b - 273.15: " + "  ".join(f"{xx:.0f} km {np.interp(xx, xtb, tb):+.3f}" for xx in
                                       (0, 100, 200, 300, 400, 500, 600, 700, 800, 850, 880, 890)))
    out = out or os.path.join(os.path.dirname(os.path.normpath(args[0])), f"extrap_head{int(round(head_t)):04d}")
    np.savez(out + ".npz", x=x * 1e3, zs=z, tb_x=xtb * 1e3, tb=tb, head_km=head_t, xgl_km=xgl_n, x_extrap_km=x_e, heads_km=heads,
             d_km=np.array(ds))
    print(f"-> {out}.npz")
    import plotly.graph_objects as go
    fig = go.Figure()
    n = len(prof)
    for i, (d, h, xgl, xs, zs) in enumerate(prof):
        g = xs <= xgl + 1e-3
        fig.add_trace(go.Scatter(x=xs[g], y=zs[g], mode="lines", name=f"d {d:g} km (head {h:.0f})",
                                 line=dict(width=1, color=f"rgba(60,90,160,{0.15 + 0.8 * i / max(n - 1, 1):.2f})"),
                                 legendgroup="pts", showlegend=i in (0, n - 1)))
    fig.add_trace(go.Scatter(x=x, y=z, mode="lines", name="the start", line=dict(width=3, color="black")))
    fig.add_trace(go.Scatter(x=xb, y=zb_, mode="lines", name=f"(the bridge {x_e:.0f} .. {head_t:g} km)",
                             line=dict(width=3, color="orange")))
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
    ft = go.Figure()
    for i, ((d, h, _, _, _), (xu, Tu)) in enumerate(zip(prof, tbp)):
        ft.add_trace(go.Scatter(x=xu, y=Tu, mode="lines", name=f"d {d:g} km (head {h:.0f})", legendgroup="pts",
                                line=dict(width=1, color=f"rgba(60,90,160,{0.15 + 0.8 * i / max(n - 1, 1):.2f})"),
                                showlegend=i in (0, n - 1)))
    ft.add_trace(go.Scatter(x=xtb, y=tb, mode="lines", name="the start's bed (heat solve Dirichlet)",
                            line=dict(width=3, color="black")))
    ft.add_vline(x=head_t, line=dict(color="red", dash="dash", width=1))
    ft.update_layout(template="plotly_white", xaxis_title="x (km)", yaxis_title="T_b - 273.15 (K)",
                     title=f"basal temperature extrapolated to the patch head at {head_t:g} km")
    ft.write_html(out + "_tb.html", include_plotlyjs="cdn")
    print(f"-> {out}_tb.html")
    print(f"-> {out}.html")


if __name__ == "__main__":
    main()
