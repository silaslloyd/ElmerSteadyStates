#!/usr/bin/env python3
"""
The ice flux across the grounding line, per frame, from frames files (plots/vtu_frames.py: the surface and the base of
every output step; no interior of the ice), and a plot of it over time.

    python3 plots/gl_flux.py FRAMES.npz [FRAMES2.npz ...] [--out NAME] [--rho 917]

The grounding line: the contour between floating (GroundedMask -1) and grounded (0 on the grounding line, 1) base
nodes. Along it, per segment: the thickness H = Zs - Zb and the depth-averaged velocity normal to the line (grounded
-> floating). The frames have only the surface and base velocities, so the depth average is estimated:
    u_mean = u_b + f (u_s - u_b)
with f = 0.8 (the shallow-ice shape for n = 3) as the estimate, and f = 0 (all at the basal speed) and f = 1 (all at
the surface speed) as hard bounds: the true depth average lies between the basal and surface speeds wherever the
speed decreases with depth. Near the grounding line the ice mostly slides, so the bounds are close.

With it, the accumulation integrated over the grounded ice: the marine model's surface accumulation is uniform
(aDIM, case/parameters.lua: 0.0321 m/yr of ice; --acc A for another), so this is a x the grounded area, i.e. a x the
grounding-line position integrated across the strip (from the divide at x = 0: the area to the left of the line,
by Green's theorem along it). In a steady state with no basal melt the grounding-line flux equals it.

Below it, the surface elevation over time at one point (--point X Y in km; default the centre of the grounded sheet
in the first frame: halfway from the divide to the grounding line, mid-strip), every frame, also in NAME_surface.csv.

--compare FILE: a second run (its own CSVs) over both panels, and the first run cut to the second's time span.

Re-running reads NAME.csv and computes only the frames not in it yet (new output of a running job); --recompute
redoes all of them. The plot opens in a window (and is saved); --no-show only saves it.

Out: NAME.csv (time, flux estimate and bounds and the accumulation in m^3/yr and Gt/yr, grounding-line length,
grounded area) and NAME.png. NAME defaults to the first file's name + "_glflux". Velocities are taken as m/yr (as
the model's).
"""
import os
import sys

import numpy as np

F_SHAPE = 0.8
A_DIM = 0.03213938607148378      # m/yr of ice: aDIM of case/parameters.lua (ascale with aND = 1)


def flux_series(paths, out, recompute, rho, acc, plt, mtri):
    """the grounding-line flux of every frame of the files (cached in OUT.csv): rows (t, q, q_low, q_high, length,
    grounded area), sorted by time"""
    rows, done = [], set()
    if os.path.exists(out + ".csv") and not recompute:       # the frames already computed (by time)
        with open(out + ".csv") as fh:
            head = fh.readline().strip().split(",")
        if head[:6] == ["time_yr", "flux_m3_per_yr", "flux_low_m3_per_yr", "flux_high_m3_per_yr", "gl_length_m",
                        "grounded_area_m2"]:
            old = np.atleast_2d(np.loadtxt(out + ".csv", delimiter=",", skiprows=1))
            rows = [tuple(r[:6]) for r in old if r.size >= 6]
            done = {round(r[0], 3) for r in rows}
            print(f"{out}.csv: {len(rows)} frames already computed (--recompute: all again)")
    for path in paths:
        d = np.load(path, allow_pickle=True)
        if all(round(float(t), 3) in done for t in d["frame_times"]):
            continue
        bt = mtri.Triangulation(d["base_x"].astype(float), d["base_y"].astype(float))
        tt = mtri.Triangulation(d["top_x"].astype(float), d["top_y"].astype(float))
        for k in [k for k, t in enumerate(d["frame_times"]) if round(float(t), 3) not in done]:
            t = d["frame_times"][k]
            done.add(round(float(t), 3))
            mask = d["base_mask"][k].astype(float)
            if mask.min() > -0.5 or mask.max() < -0.5:
                rows.append((float(t), np.nan, np.nan, np.nan, 0.0, np.nan))
                continue
            # the grounding line: the -0.5 contour of the mask (between floating and grounding-line/grounded nodes)
            fig = plt.figure()
            cs = plt.tricontour(bt, mask, levels=[-0.5])
            segs = [p for p in cs.allsegs[0] if len(p) > 1]
            plt.close(fig)
            if not segs:
                rows.append((float(t), np.nan, np.nan, np.nan, 0.0, np.nan))
                continue
            P = np.concatenate([np.c_[s[:-1], s[1:]] for s in segs])      # x0 y0 x1 y1 per segment
            mid = 0.5 * (P[:, :2] + P[:, 2:])
            dl = P[:, 2:] - P[:, :2]
            L = np.hypot(dl[:, 0], dl[:, 1])
            ok = L > 0
            P, mid, dl, L = P[ok], mid[ok], dl[ok], L[ok]
            nrm = np.c_[dl[:, 1], -dl[:, 0]] / L[:, None]                  # a unit normal to each segment
            # orient each normal from grounded to floating: the mask falls along it
            fm = mtri.LinearTriInterpolator(bt, mask)
            eps = 50.0
            ahead = np.asarray(fm(mid[:, 0] + eps * nrm[:, 0], mid[:, 1] + eps * nrm[:, 1]).filled(np.nan))
            back = np.asarray(fm(mid[:, 0] - eps * nrm[:, 0], mid[:, 1] - eps * nrm[:, 1]).filled(np.nan))
            flip = np.where(np.isfinite(ahead) & np.isfinite(back), ahead > back, False)
            nrm[flip] *= -1.0

            def at(tri, f):
                return np.asarray(mtri.LinearTriInterpolator(tri, f.astype(float))(mid[:, 0], mid[:, 1]).filled(np.nan))
            H = at(tt, d["top_z"][k]) - at(bt, d["base_z"][k])
            us = at(tt, d["top_vx"][k]) * nrm[:, 0] + at(tt, d["top_vy"][k]) * nrm[:, 1]
            ub = at(bt, d["base_vx"][k]) * nrm[:, 0] + at(bt, d["base_vy"][k]) * nrm[:, 1]
            good = np.isfinite(H) & np.isfinite(us) & np.isfinite(ub)
            q = lambda f: float(np.sum((H * (ub + f * (us - ub)) * L)[good]))
            lo, hi = sorted((q(0.0), q(1.0)))
            # the grounded area: the area between the divide (x = 0) and the line, |sum of x dy| along it
            area = abs(float(np.sum(mid[:, 0] * dl[:, 1])))
            rows.append((float(t), q(F_SHAPE), lo, hi, float(L[good].sum()), area))
            print(f"  {os.path.basename(path)} t {t:9.1f} yr: grounding-line flux {q(F_SHAPE):.4e} m^3/yr "
                  f"({q(F_SHAPE) * rho / 1e12:.4f} Gt/yr; bounds {lo * rho / 1e12:.4f} .. {hi * rho / 1e12:.4f}), "
                  f"line length {L[good].sum() / 1e3:.1f} km; accumulation upstream {acc * area * rho / 1e12:.4f} Gt/yr "
                  f"(grounded area {area / 1e6:.0f} km^2)", flush=True)
    rows.sort()
    R = np.array(rows)
    accq = acc * R[:, 5]                                   # m^3/yr of ice
    hdr = ("time_yr,flux_m3_per_yr,flux_low_m3_per_yr,flux_high_m3_per_yr,gl_length_m,grounded_area_m2,"
           "accumulation_m3_per_yr,flux_Gt_per_yr,flux_low_Gt,flux_high_Gt,accumulation_Gt_per_yr")
    np.savetxt(out + ".csv", np.c_[R, accq, R[:, 1:4] * rho / 1e12, accq * rho / 1e12], delimiter=",", header=hdr,
               comments="")
    return R, accq


def default_point(path):
    """the centre of the grounded sheet in the first frame: halfway from the divide to the grounding line, mid-strip"""
    D0 = np.load(path, allow_pickle=True)
    bx0, m0 = D0["base_x"], D0["base_mask"][0]
    gx = float(bx0[m0 > -0.5].max()) if (m0 > -0.5).any() else float(bx0.max())
    return (0.5 * gx / 1e3, 0.5 * float(D0["top_y"].min() + D0["top_y"].max()) / 1e3)


def surface_series(paths, point, out):
    """the surface elevation at POINT (km) in every frame of the files (linear within the mesh), sorted by time"""
    from scipy.spatial import Delaunay
    ts, zs = [], []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        tri = Delaunay(np.c_[d["top_x"], d["top_y"]])
        P = np.array([[point[0] * 1e3, point[1] * 1e3]])
        j = int(tri.find_simplex(P)[0])
        if j < 0:
            print(f"  {os.path.basename(path)}: the point {point} km is outside the surface mesh")
            continue
        T = tri.transform[j]
        b = T[:2] @ (P[0] - T[2]); w = np.r_[b, 1 - b.sum()]          # linear (barycentric) weights
        nodes = tri.simplices[j]
        ts += list(d["frame_times"]); zs += list(d["top_z"][:, nodes].astype(float) @ w)
    ts, first = np.unique(np.round(np.array(ts, float), 3), return_index=True)   # sorted, each time once
    zs = np.array(zs)[first]
    np.savetxt(out + "_surface.csv", np.c_[ts, zs], delimiter=",",
               header=f"time_yr,surface_m_at_x{point[0]:g}_y{point[1]:g}_km", comments="")
    print(f"surface elevation at x {point[0]:.1f} y {point[1]:.1f} km: {zs.min():.1f} .. {zs.max():.1f} m "
          f"({len(ts)} frames) -> {out}_surface.csv")
    return ts, zs


def main():
    args = sys.argv[1:]
    out, rho, acc = None, 917.0, A_DIM
    if "--out" in args:
        k = args.index("--out"); out = args[k + 1]; args = args[:k] + args[k + 2:]
    recompute = "--recompute" in args
    show = "--no-show" not in args                          # --no-show: only save the image (e.g. on tethys)
    args = [a for a in args if a not in ("--recompute", "--no-show")]
    if "--acc" in args:
        k = args.index("--acc"); acc = float(args[k + 1]); args = args[:k] + args[k + 2:]
    point = None
    if "--point" in args:                                   # --point X Y (km): where to follow the surface elevation
        k = args.index("--point"); point = (float(args[k + 1]), float(args[k + 2])); args = args[:k] + args[k + 3:]
    if "--rho" in args:
        k = args.index("--rho"); rho = float(args[k + 1]); args = args[:k] + args[k + 2:]
    compare = None
    if "--compare" in args:                                 # --compare FILE: a second run over the same plots
        k = args.index("--compare"); compare = args[k + 1]; args = args[:k] + args[k + 2:]
    if not args:
        raise SystemExit(__doc__)
    out = out or os.path.splitext(args[0])[0] + "_glflux"

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    G = 1e12 / rho                                          # m^3/yr of ice per Gt/yr
    R, accq = flux_series(args, out, recompute, rho, acc, plt, mtri)
    point = point or default_point(args[0])
    ts, zs = surface_series(args, point, out)
    fig, (ax, az) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    lab1 = os.path.basename(args[0])[:-4].replace("_frames", "") if compare else None
    if compare:                                             # the second run, and only its time span of the first
        out2 = os.path.splitext(compare)[0] + "_glflux"
        R2, accq2 = flux_series([compare], out2, recompute, rho, acc, plt, mtri)
        ts2, zs2 = surface_series([compare], point, out2)
        t0, t1 = R2[:, 0].min() - 1e-6, R2[:, 0].max() + 1e-6
        m = (R[:, 0] >= t0) & (R[:, 0] <= t1)
        R, accq = R[m], accq[m]
        m = (ts >= t0) & (ts <= t1)
        ts, zs = ts[m], zs[m]
        lab2 = os.path.basename(compare)[:-4].replace("_frames", "")
    ax.fill_between(R[:, 0], R[:, 2] / G, R[:, 3] / G, color="tab:blue", alpha=0.25,
                    label="bounds (basal .. surface speed)" + (f", {lab1}" if lab1 else ""))
    ax.plot(R[:, 0], R[:, 1] / G, "o-", color="tab:blue", ms=3,
            label=f"flux estimate (u_b + {F_SHAPE:g} (u_s - u_b))" + (f", {lab1}" if lab1 else ""))
    ax.plot(R[:, 0], accq / G, "s--", color="tab:orange", ms=3,
            label=f"accumulation upstream (a x grounded area, a = {acc:.4g} m/yr)" + (f", {lab1}" if lab1 else ""))
    az.plot(ts, zs, "o-", color="tab:green", ms=3, label=lab1)
    if compare:
        ax.fill_between(R2[:, 0], R2[:, 2] / G, R2[:, 3] / G, color="tab:red", alpha=0.15)
        ax.plot(R2[:, 0], R2[:, 1] / G, "D-", color="tab:red", ms=4, label=f"flux estimate, {lab2}")
        ax.plot(R2[:, 0], accq2 / G, "D--", color="tab:brown", ms=4, label=f"accumulation upstream, {lab2}")
        az.plot(ts2, zs2, "D-", color="tab:red", ms=4, label=lab2)
        az.legend()
    ax.set_ylabel("Gt/yr")
    ax.grid(True, alpha=0.4); ax.legend(fontsize=8)
    az.set_ylabel(f"surface elevation (m)\nat x {point[0]:.0f}, y {point[1]:.0f} km")
    az.set_xlabel("time (yr)"); az.grid(True, alpha=0.4)
    fig.tight_layout()
    png = out + (f"_vs_{os.path.basename(compare)[:-4]}" if compare else "") + ".png"
    fig.savefig(png, dpi=200)
    print(f"{out}.csv, {png}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
