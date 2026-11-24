#!/usr/bin/env python3
"""A continental shelf break in the bed of an inverted-sector state: the bed (frozen "bedrock", every node: the mesh
is mapped from it at the start of each run, so nothing else changes) bends smoothly down to a slope FACTOR times the
bed's interior slope m, downstream of the radius R_BREAK (the ice flows inwards: downstream = smaller radius):

    b(r) = b_old(r) - (FACTOR m - s0) w softplus((R_BREAK - r)/w),   softplus(u) = log(1 + exp(u))

m: the linear fit of the old bed over --fit-r-km (default 800 .. 1450 km: the sloping interior; the bed of the
inv38_cut_ext states is flat at -224.1 m from r ~ 705 km to the front, where the flowline it came from ended);
s0: the old bed's own slope around R_BREAK (0 on the flat part). The slope goes from s0 to FACTOR m over a few w
around R_BREAK; the old bed is kept as it is and only this bend added, so the bed under the grounded ice upstream
changes by centimetres at most (and their Zb with it). Upstream of --keep-above-km (default R_BREAK + 10 w, where the
bend is below 1 cm) the bed is exactly the old one.

    python3 jobs/shelf_break.py IN.npz OUT.npz [--r-break-km 450] [--factor 10] [--width-km 10] [--keep-above-km R]

Checks: the bed under grounded ice (GroundedMask > -0.5) changes by under 1 m, and every floating base stays above the
new bed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import find_name, load_state, save_state  # noqa: E402


def lookup(rec, name, gids):
    g, v = rec[find_name(rec, name)]
    j = np.clip(np.searchsorted(g, gids), 0, len(g) - 1)
    if not np.array_equal(g[j], gids):
        raise SystemExit(f"'{name}' does not cover all the bedrock nodes")
    return v[j]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--r-break-km", type=float, default=450.0)
    ap.add_argument("--factor", type=float, default=10.0)
    ap.add_argument("--width-km", type=float, default=10.0)
    ap.add_argument("--keep-above-km", type=float, default=None)
    ap.add_argument("--fit-r-km", type=float, nargs=2, default=(800.0, 1450.0))
    ap.add_argument("--keep-flat", dest="continue_slope", action="store_false",
                    help="keep a flat outer stretch as it is (by default the bed wherever it lies above the "
                         "interior's linear fit, i.e. the flat -224.1 m stretch the extended domain got from the shorter "
                         "flowline, is put on that line, all the way to the front, and the bend added on top)")
    ap.add_argument("--strip-divide-r-km", type=float, default=None,
                    help="a periodic strip (x from the divide): r = this - x, the radius of the sector it stands for "
                         "(1520.8 for configs/coarse_rect_ext.toml, as coarse_sector_inv38_cut_ext), so --r-break-km "
                         "and --fit-r-km mean the same as on the sector")
    ap.add_argument("--force", action="store_true",
                    help="write even if the bed under grounded ice changes (e.g. a bed template for jobs/thin_start.py)")
    a = ap.parse_args()
    s = load_state(a.src)
    allf = {**s.diag, **s.frozen}
    kb = find_name(s.frozen, "bedrock")
    g, bed = s.frozen[kb]
    x, y = lookup(allf, "coordinate 1", g), lookup(allf, "coordinate 2", g)
    # a periodic strip (a sector of infinite radius): the radius of the sector it stands for, divide r D at x = 0
    r = (a.strip_divide_r_km * 1e3 - x) if a.strip_divide_r_km else np.hypot(x, y)
    rb, w = a.r_break_km * 1e3, a.width_km * 1e3
    keep = (a.keep_above_km * 1e3) if a.keep_above_km is not None else rb + 10 * w
    fit = (r > a.fit_r_km[0] * 1e3) & (r < a.fit_r_km[1] * 1e3)
    m, c0 = np.polyfit(r[fit], bed[fit], 1)
    res = bed[fit] - (c0 + m * r[fit])
    print(f"{a.src}: {len(g)} nodes; bed fit over r {a.fit_r_km[0]:g} .. {a.fit_r_km[1]:g} km: "
          f"{c0:.2f} + {m * 1e3:.5f} m/km x r (rms misfit {np.sqrt(np.mean(res ** 2)):.3f} m)")
    if a.continue_slope:
        line = c0 + m * r
        above = bed > line + 1e-6
        bed0 = bed.copy()
        bed = np.minimum(bed, line)
        print(f"continue the slope: {int(above.sum())} nodes above the line lowered onto it (by up to "
              f"{np.max(bed0 - bed):.1f} m; from r = {r[above].max() / 1e3 if above.any() else np.nan:.1f} km down)")
        keep = np.inf                                  # the bend over the whole (now sloping) outer part
    near = np.abs(r - rb) < 3 * w
    s0 = np.polyfit(r[near], bed[near], 1)[0] if near.sum() > 10 else m
    add = a.factor * m - s0
    print(f"old bed's slope around r = {rb / 1e3:g} km: {s0 * 1e3:.4f} m/km; the bend adds {add * 1e3:.4f} m/km")
    u = (rb - r) / w
    sp = np.where(u > 30, u, np.log1p(np.exp(np.minimum(u, 30))))
    new = np.where(r < keep, bed - add * w * sp, bed)
    if a.continue_slope:
        bed = bed0                                     # the changes (and the bases on the bed) against the original
    d = new - bed
    ch = np.abs(d) > 1e-6
    print(f"bed: slope {s0 * 1e3:.4f} m/km -> {a.factor * m * 1e3:.4f} m/km downstream of r = {rb / 1e3:g} km "
          f"(width {w / 1e3:g} km); changed below r = {keep / 1e3:g} km: {ch.sum()} nodes, by {d.min():.1f} .. "
          f"{d.max():.1f} m")
    for rk in (keep / 1e3, rb / 1e3 + 20, rb / 1e3, rb / 1e3 - 20, r.min() / 1e3):
        k = np.argmin(np.abs(r - rk * 1e3))
        print(f"  r {r[k] / 1e3:7.1f} km: bed {bed[k]:9.1f} -> {new[k]:9.1f} m")
    # checks: grounded nodes where the bed changes; floating bases below the new bed
    ok = True
    try:
        gg, gv = allf[find_name(allf, "groundedmask")]
        jg = np.clip(np.searchsorted(g, gg), 0, len(g) - 1)
        okg = g[jg] == gg
        gm = np.full(len(g), -1.0)
        gm[jg[okg]] = gv[okg]
        print(f"  GroundedMask on {okg.sum()} of the nodes")
        gch = ch & (gm > -0.5)
        print(f"  grounding line (GroundedMask > -0.5) down to r = {r[gm > -0.5].min() / 1e3:.1f} km; the bed under "
              f"grounded ice changes by at most {np.abs(d[gch]).max() if gch.any() else 0.0:.3f} m (their Zb with it)")
        if gch.any() and np.abs(d[gch]).max() > 1.0:
            ok = False
            print("PROBLEM: the grounded ice reaches where the bed changes by more than 1 m: a smaller --r-break-km "
                  "or an earlier state")
    except KeyError as e:
        ok = False
        print(f"PROBLEM: no GroundedMask to check the grounded ice against: {e}")
    blocks = s.layout.split(s.x)
    xs = s.x.copy()
    zbn = next((b.name for b in s.layout.blocks if b.name.lower() == "zb"), None)
    if zbn is not None:
        sl = s.layout.slices()[zbn]
        gz = next(b.gids for b in s.layout.blocks if b.name == zbn)
        zb = blocks[zbn].copy()
        j = np.clip(np.searchsorted(g, gz), 0, len(g) - 1)
        if np.array_equal(g[j], gz):
            on = np.abs(zb - bed[j]) < 1e-3            # the base on the bed (grounded): it moves with the bed
            zb[on] = new[j][on]
            xs[sl] = zb
            low = ch[j] & ~on & (zb < new[j] + 1.0)
            if low.any():
                ok = False
                print(f"PROBLEM: {low.sum()} floating bases within 1 m of the new bed or below it")
                xj, rj = x[j], r[j]
                for q in np.where(low)[0][:12]:
                    print(f"    x {xj[q] / 1e3:9.3f} km (r {rj[q] / 1e3:8.2f} km): base {zb[q]:9.2f} m, old bed {bed[j][q]:9.2f} m, "
                          f"new bed {new[j][q]:9.2f} m, grounded mask {gm[j][q]:.0f}")
            else:
                gap = (zb - new[j])[ch[j] & ~on]
                print(f"  {on[ch[j]].sum()} bases on the bed moved with it; the floating ones stay above it "
                      f"(smallest gap {gap.min() if len(gap) else np.nan:.1f} m)")
    if not ok and not a.force:
        raise SystemExit("not written (--force: write anyway)")
    frozen = dict(s.frozen)
    frozen[kb] = (g, new)
    save_state(a.out, s.layout, xs, s.p, frozen, s.diag, **{k: v for k, v in s.meta.items()},
               shelf_break=f"r {a.r_break_km:g} km, x{a.factor:g}, w {a.width_km:g} km, from {a.src}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
