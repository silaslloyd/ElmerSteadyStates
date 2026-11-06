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

Re-running reads NAME.csv and computes only the frames not in it yet (new output of a running job); --recompute
redoes all of them.

Out: NAME.csv (time, flux estimate and bounds and the accumulation in m^3/yr and Gt/yr, grounding-line length,
grounded area) and NAME.png. NAME defaults to the first file's name + "_glflux". Velocities are taken as m/yr (as
the model's).
"""
import os
import sys

import numpy as np

F_SHAPE = 0.8
A_DIM = 0.03213938607148378      # m/yr of ice: aDIM of case/parameters.lua (ascale with aND = 1)


def main():
    args = sys.argv[1:]
    out, rho, acc = None, 917.0, A_DIM
    if "--out" in args:
        k = args.index("--out"); out = args[k + 1]; args = args[:k] + args[k + 2:]
    recompute = "--recompute" in args
    args = [a for a in args if a != "--recompute"]
    if "--acc" in args:
        k = args.index("--acc"); acc = float(args[k + 1]); args = args[:k] + args[k + 2:]
    if "--rho" in args:
        k = args.index("--rho"); rho = float(args[k + 1]); args = args[:k] + args[k + 2:]
    if not args:
        raise SystemExit(__doc__)
    out = out or os.path.splitext(args[0])[0] + "_glflux"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

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
    for path in args:
        d = np.load(path, allow_pickle=True)
        bx, by = d["base_x"].astype(float), d["base_y"].astype(float)
        tx, ty = d["top_x"].astype(float), d["top_y"].astype(float)
        if all(round(float(t), 3) in done for t in d["frame_times"]):
            continue
        bt = mtri.Triangulation(bx, by)
        tt = mtri.Triangulation(tx, ty)
        todo = [k for k, t in enumerate(d["frame_times"]) if round(float(t), 3) not in done]
        if not todo:
            continue
        for k in todo:
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
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(R[:, 0], R[:, 2] * rho / 1e12, R[:, 3] * rho / 1e12, color="tab:blue", alpha=0.25,
                    label="bounds (basal .. surface speed)")
    ax.plot(R[:, 0], R[:, 1] * rho / 1e12, "o-", color="tab:blue", ms=3,
            label=f"estimate (u_b + {F_SHAPE:g} (u_s - u_b))")
    ax.plot(R[:, 0], accq * rho / 1e12, "s--", color="tab:orange", ms=3,
            label=f"accumulation upstream (a x grounded area, a = {acc:.4g} m/yr)")
    ax.set_xlabel("time (yr)"); ax.set_ylabel("Gt/yr")
    ax.grid(True, alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200)
    print(f"{out}.csv, {out}.png")


if __name__ == "__main__":
    main()
