#!/usr/bin/env python3
"""The water sheet's forcing across the grounding line, at its worst spot: a run's base VTUs (THERMAL=elmerstreams,
base fields Phi0 Output, Szz Output, zb), the nodes in a narrow band along the radius through the node of largest
sheet (or --at R_KM ANGLE_DEG), sorted by radius: grounded mask, cold/temperate mask, bed, sigma_zz, Phi0, the sheet.
An element-scale zigzag of Phi0 next to the grounding line (wells and ridges ~1 km apart) drives the paired sheet
highs and lows seen there.

    python3 plots/sheet_gl_profile.py RUNDIR [--frame K] [--at R_KM ANGLE_DEG] [--half-km 20] [--band-m 600]

--frame: the output (default the last); --half-km: the radius range either side (default 20 km); --band-m: the band's
half width across (default 600 m). Also: the hydrostatic Phi0 rhoi g H + rhow g zb is not available here (no Zs on
the base), so the roughness is given as the second difference of Phi0 along the band against its first difference.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vtu_frames import read_vtu  # noqa: E402


def opt(args, name, n=1, conv=float, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = [conv(a) for a in args[k + 1:k + 1 + n]]
    return args[:k] + args[k + 1 + n:], (v[0] if n == 1 else v)


def main():
    args = sys.argv[1:]
    args, frame = opt(args, "--frame", conv=int)
    args, at = opt(args, "--at", n=2)
    args, half = opt(args, "--half-km", default=20.0)
    args, band = opt(args, "--band-m", default=600.0)
    if len(args) != 1:
        raise SystemExit(__doc__)
    files = sorted(glob.glob(os.path.join(args[0], "base_t*.pvtu")))
    if not files:
        raise SystemExit(f"no base_t*.pvtu in {args[0]}")
    f = files[-1] if frame is None else files[frame - 1]
    X, D = read_vtu(f)
    k = {n.lower(): n for n in D}
    need = ["water sheet thickness", "phi0 output", "szz output"]
    miss = [n for n in need if n not in k]
    if miss:
        raise SystemExit(f"{f}: no {', '.join(miss)} (a run before the Phi0 output was added?): {sorted(D)}")
    col = lambda n: D[k[n]][:, 0] if n in k else np.full(len(X), np.nan)
    h, phi, szz = col("water sheet thickness"), col("phi0 output"), col("szz output")
    gm, ct = col("groundedmask"), col("coldtempmask")
    zb = col("zb") if "zb" in k else col("bedrock")
    r = np.hypot(X[:, 0], X[:, 1])
    th = np.arctan2(X[:, 1], X[:, 0])
    if at is None:
        i0 = int(np.nanargmax(h))
        r0, th0 = r[i0], th[i0]
    else:
        r0, th0 = at[0] * 1e3, np.radians(at[1])
    across = r * np.sin(th - th0)
    sel = np.where((np.abs(across) < band) & (np.abs(r - r0) < half * 1e3) & (np.cos(th - th0) > 0))[0]
    sel = sel[np.argsort(r[sel])]
    print(f"{f}: band through r {r0 / 1e3:.1f} km, angle {np.degrees(th0):.3f} deg; {len(sel)} nodes "
          f"(|across| < {band:.0f} m, r +- {half:.0f} km)")
    print(f"{'r (km)':>8} {'across':>7} {'grnd':>5} {'temp':>5} {'zb (m)':>8} {'szz (MPa)':>10} {'Phi0 (MPa)':>11} "
          f"{'h (m)':>9}")
    for i in sel:
        print(f"{r[i] / 1e3:8.2f} {across[i]:7.0f} {gm[i]:5.0f} {ct[i]:5.0f} {zb[i]:8.1f} {szz[i]:10.4f} "
              f"{phi[i]:11.4f} {h[i]:9.3g}")
    if len(sel) > 4:
        rr, pp = r[sel], phi[sel]
        d1 = np.diff(pp) / np.maximum(np.diff(rr), 1.0)
        sign = np.sign(d1[np.abs(d1) > 0])
        flips = int(np.sum(sign[1:] != sign[:-1]))
        print(f"Phi0 along the band: {flips} changes of slope sign over {len(d1)} intervals; "
              f"slope {np.nanmin(d1) * 1e3:.4f} .. {np.nanmax(d1) * 1e3:.4f} MPa/km "
              f"(a smooth potential: no or few sign changes)")


if __name__ == "__main__":
    main()
