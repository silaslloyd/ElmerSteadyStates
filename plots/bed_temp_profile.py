#!/usr/bin/env python3
"""The basal temperature of saved states against the distance from the divide, in bins: how close the bed is to the
pressure melting point (subtemperate sliding grows within a few K of it) and where the cold/temperate mask is
temperate. Strips by x, sectors by the distance from the divide (r, or the outer radius - r for an inverted sector).

    python3 plots/bed_temp_profile.py STATE.npz [STATE.npz ...] [--bin 50] [--beta 8.7e-4]

Per bin (km from the divide): bed nodes, mean and max basal T (K), mean and max Tb - Tpmp (Tpmp = 273.15 - beta H,
beta K per m of ice), the fraction within 1 K and 3 K of Tpmp, the fraction ColdTempMask temperate (> 0) and grounded.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import find_name, load_state   # noqa: E402
from ess.transfer import _all_fields, _block, _lookup   # noqa: E402


def profile(path, binkm, beta):
    s = load_state(path)
    f = _all_fields(s)
    gT, T = _block(s, "Temperature")
    x = _lookup(f, "coordinate 1", gT)
    y = _lookup(f, "coordinate 2", gT)
    z = _lookup(f, "coordinate 3", gT)
    sector = np.min(y) < -1.0
    if sector:
        r = np.hypot(x, y)
        inverted = None
        dist = r
    else:
        dist = x
    # the base and the surface of each column: columns by rounded (x, y)
    key = np.round(x, 0) * 1e7 + np.round(y, 0)
    gz, Zs = _block(s, "Zs")
    zsx = _lookup(f, "coordinate 1", gz); zsy = _lookup(f, "coordinate 2", gz)
    zs_of = dict(zip(np.round(zsx, 0) * 1e7 + np.round(zsy, 0), Zs))
    if any(b.name.lower() == "zb" for b in s.layout.blocks):
        gb, Zb = _block(s, "Zb")
        bx = _lookup(f, "coordinate 1", gb); by = _lookup(f, "coordinate 2", gb)
        zb_of = dict(zip(np.round(bx, 0) * 1e7 + np.round(by, 0), Zb))
    else:
        gb, Bd = f[find_name(f, "bedrock")]
        bx = _lookup(f, "coordinate 1", gb); by = _lookup(f, "coordinate 2", gb)
        zb_of = dict(zip(np.round(bx, 0) * 1e7 + np.round(by, 0), Bd))
    zb = np.array([zb_of.get(k, np.nan) for k in key])
    zsn = np.array([zs_of.get(k, np.nan) for k in key])
    bed = np.abs(z - zb) < 1.0                     # the ice node on the base of its column
    uk, first = np.unique(key[bed], return_index=True)
    idx = np.where(bed)[0][first]
    H = zsn[idx] - zb[idx]
    Tb = T[idx]
    D = dist[idx]
    if sector:
        rmax = float(np.max(dist))
        # inverted when the ice is thicker at the outer arc (the divide there)
        o = np.argsort(D)
        inverted = np.nanmean(H[o[-len(o) // 20:]]) > np.nanmean(H[o[:len(o) // 20]])
        D = (rmax - D) if inverted else D
    Tp = 273.15 - beta * H
    dT = Tb - Tp
    ct = gm = None
    for name in ("ColdTempMask", "GroundedMask"):
        try:
            g, v = s.frozen[find_name(s.frozen, name)]
        except KeyError:
            continue
        gx = _lookup(f, "coordinate 1", g); gy = _lookup(f, "coordinate 2", g)
        m = dict(zip(np.round(gx, 0) * 1e7 + np.round(gy, 0), v))
        arr = np.array([m.get(k, np.nan) for k in uk])
        if name == "ColdTempMask":
            ct = arr
        else:
            gm = arr
    print(f"\n{path}: {'sector' + (' (inverted)' if inverted else '') if sector else 'strip'}, {len(idx)} bed nodes; "
          f"temperate mask {int(np.nansum(ct > 0)) if ct is not None else '-'}, Tb - Tpmp max {np.nanmax(dT):.2f} K")
    print(f"{'s (km)':>10} {'nodes':>6} {'Tb mean':>8} {'Tb max':>8} {'dT mean':>8} {'dT max':>7} {'<1K':>6} {'<3K':>6} "
          f"{'temp.':>6} {'grnd':>6}")
    edges = np.arange(0, np.nanmax(D) / 1e3 + binkm, binkm)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (D / 1e3 >= a) & (D / 1e3 < b)
        if not m.any():
            continue
        tf = f"{np.mean(ct[m] > 0):6.2f}" if ct is not None else "     -"
        gf = f"{np.mean(gm[m] > -0.5):6.2f}" if gm is not None else "     -"
        print(f"{a:4.0f}-{b:<4.0f} {m.sum():6d} {np.nanmean(Tb[m]):8.2f} {np.nanmax(Tb[m]):8.2f} {np.nanmean(dT[m]):8.2f} "
              f"{np.nanmax(dT[m]):7.2f} {np.mean(dT[m] > -1):6.2f} {np.mean(dT[m] > -3):6.2f} {tf} {gf}")


def main():
    args = sys.argv[1:]
    binkm, beta = 50.0, 8.7e-4
    for o in ("--bin", "--beta"):
        if o in args:
            k = args.index(o)
            v = float(args[k + 1])
            args = args[:k] + args[k + 2:]
            binkm, beta = (v, beta) if o == "--bin" else (binkm, v)
    if not args:
        raise SystemExit(__doc__)
    for p in args:
        profile(p, binkm, beta)


if __name__ == "__main__":
    main()
