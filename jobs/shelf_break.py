#!/usr/bin/env python3
"""A continental shelf break in the bed of an inverted-sector state: the bed (frozen "bedrock", every node: the mesh
is mapped from it at the start of each run, so nothing else changes) turns smoothly from its linear slope to FACTOR
times that slope, downstream of the radius R_BREAK (the ice flows inwards: downstream = smaller radius):

    b(r) = lin(r) - (FACTOR - 1) m w softplus((R_BREAK - r)/w),   softplus(u) = log(1 + exp(u))

lin(r) = a + m r is the linear fit of the state's bed over the grounded interior (fit_r range); the slope goes from
m to FACTOR m over a few w around R_BREAK (half the extra slope at R_BREAK itself). Upstream of --keep-above-km (default
R_BREAK + 10 w) the bed is left exactly as it is (the correction there is below 1 cm); below it the bed is the new
curve (min(old, line): also deepening the flat bed beyond the end of ElmerStreams' tables, s > 1020 km, r < 500 km).
Grounded bases (Zb on the bed) move with the bed (by centimetres where the grounding line is ~50 km from the break).

    python3 jobs/shelf_break.py IN.npz OUT.npz [--r-break-km 450] [--factor 10] [--width-km 10] [--keep-above-km R]

Checks: no grounded node (GroundedMask > -0.5) where the bed changes, and every floating base (Zb) above the new bed.
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
    ap.add_argument("--fit-r-km", type=float, nargs=2, default=(600.0, 1400.0))
    a = ap.parse_args()
    s = load_state(a.src)
    allf = {**s.diag, **s.frozen}
    kb = find_name(s.frozen, "bedrock")
    g, bed = s.frozen[kb]
    x, y = lookup(allf, "coordinate 1", g), lookup(allf, "coordinate 2", g)
    r = np.hypot(x, y)
    rb, w = a.r_break_km * 1e3, a.width_km * 1e3
    keep = (a.keep_above_km * 1e3) if a.keep_above_km is not None else rb + 10 * w
    fit = (r > a.fit_r_km[0] * 1e3) & (r < a.fit_r_km[1] * 1e3)
    m, c0 = np.polyfit(r[fit], bed[fit], 1)
    res = bed[fit] - (c0 + m * r[fit])
    print(f"{a.src}: {len(g)} nodes; bed fit over r {a.fit_r_km[0]:g} .. {a.fit_r_km[1]:g} km: "
          f"{c0:.2f} + {m * 1e3:.5f} m/km x r (rms misfit {np.sqrt(np.mean(res ** 2)):.3f} m)")
    u = (rb - r) / w
    sp = np.where(u > 30, u, np.log1p(np.exp(np.minimum(u, 30))))
    # the old bed where it follows the line, the line where the old bed stops deepening (the flat end of the tables)
    curve = np.minimum(bed, c0 + m * r) - (a.factor - 1.0) * m * w * sp
    new = np.where(r < keep, curve, bed)
    d = new - bed
    ch = np.abs(d) > 1e-6
    print(f"bed: slope {m * 1e3:.4f} m/km -> {a.factor * m * 1e3:.4f} m/km downstream of r = {rb / 1e3:g} km "
          f"(width {w / 1e3:g} km); changed below r = {keep / 1e3:g} km: {ch.sum()} nodes, by {d.min():.1f} .. "
          f"{d.max():.1f} m")
    for rk in (keep / 1e3, rb / 1e3 + 20, rb / 1e3, rb / 1e3 - 20, r.min() / 1e3):
        k = np.argmin(np.abs(r - rk * 1e3))
        print(f"  r {r[k] / 1e3:7.1f} km: bed {bed[k]:9.1f} -> {new[k]:9.1f} m")
    # checks: grounded nodes where the bed changes; floating bases below the new bed
    ok = True
    try:
        gm = lookup(allf, "groundedmask", g)
        gch = ch & (gm > -0.5)
        print(f"  grounding line (GroundedMask > -0.5) down to r = {r[gm > -0.5].min() / 1e3:.1f} km; the bed under "
              f"grounded ice changes by at most {np.abs(d[gch]).max() if gch.any() else 0.0:.3f} m (their Zb with it)")
        if gch.any() and np.abs(d[gch]).max() > 1.0:
            ok = False
            print("PROBLEM: the grounded ice reaches where the bed changes by more than 1 m: a smaller --r-break-km "
                  "or an earlier state")
    except (KeyError, SystemExit) as e:
        print(f"  (no GroundedMask check: {e})")
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
            else:
                gap = (zb - new[j])[ch[j] & ~on]
                print(f"  {on[ch[j]].sum()} bases on the bed moved with it; the floating ones stay above it "
                      f"(smallest gap {gap.min() if len(gap) else np.nan:.1f} m)")
    if not ok:
        raise SystemExit("not written")
    frozen = dict(s.frozen)
    frozen[kb] = (g, new)
    save_state(a.out, s.layout, xs, s.p, frozen, s.diag, **{k: v for k, v in s.meta.items()},
               shelf_break=f"r {a.r_break_km:g} km, x{a.factor:g}, w {a.width_km:g} km, from {a.src}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
