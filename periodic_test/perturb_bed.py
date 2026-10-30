#!/usr/bin/env python3
"""A wavy-grounding-line start for the 3D marine tests: the bed lowered by up to --amp m in a pattern across the
width, amp (1 + sin(2 pi y / W)) / 2, over a band around the grounding line (full amplitude from --up km upstream to
--down km downstream of it, cosine tapers of --taper km on both sides).

The ice keeps its thickness: where it stays grounded, its base and surface go down with the bed; where it is then
thinner than flotation, it is floated (base at -rho_i/rho_w H, surface at (1 - rho_i/rho_w) H, GroundedMask -1), so
the grounding line retreats at once in the deeper rows. Floating ice is unchanged (deeper water under it).
Everything else (temperature, the velocity and pressure for a warm start, the other frozen fields) is kept.

  python3 periodic_test/perturb_bed.py IN.npz OUT.npz [--amp 150] [--up 15] [--down 10] [--taper 15] [--width 50000]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import find_name, load_state, save_state   # noqa: E402
from ess.case import eval_lua_parameters                         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--amp", type=float, default=150.0, help="m: the largest lowering")
    ap.add_argument("--up", type=float, default=15.0, help="km upstream of the grounding line at full amplitude")
    ap.add_argument("--down", type=float, default=10.0, help="km downstream of it at full amplitude")
    ap.add_argument("--taper", type=float, default=15.0, help="km: cosine tapers beyond that band")
    ap.add_argument("--width", type=float, default=50000.0, help="m: the domain width (the pattern's wavelength)")
    ap.add_argument("--params", default="case/parameters.lua")
    a = ap.parse_args()

    P = eval_lua_parameters(a.params)
    r = P["rhoi"] / P["rhow"]
    s = load_state(a.src)
    allf = {**s.diag, **s.frozen}
    gc, X = allf[find_name(allf, "coordinate 1")]
    Y = dict(zip(*allf[find_name(allf, "coordinate 2")]))
    Xd = dict(zip(gc.tolist(), X.tolist()))
    col = lambda g: (round(Xd[g], 1), round(Y[g], 1))

    gmk = find_name(s.frozen, "GroundedMask")
    gg, gm = s.frozen[gmk]
    gm = np.array(gm, float)
    # the grounding line: the most downstream grounded interface node of the y = 0 row ... per row
    rows = {}
    for g, m in zip(gg.tolist(), gm.tolist()):
        if g in Xd:
            rows.setdefault(round(Y[g], 1), []).append((Xd[g], m))
    gl = {}
    for y, lst in rows.items():
        lst.sort()
        fl = [i for i, (_, m) in enumerate(lst) if m < -0.5]
        gl[y] = lst[fl[0] - 1][0] if fl and fl[0] > 0 else lst[-1][0]
    xgl = float(np.median(list(gl.values())))
    print(f"grounding line (median over rows) {xgl / 1e3:.2f} km, {len(rows)} rows")

    def dz(x, y):
        """the lowering (>= 0, m) at (x, y)"""
        lo, hi, t = xgl - a.up * 1e3, xgl + a.down * 1e3, a.taper * 1e3
        if x < lo - t or x > hi + t:
            w = 0.0
        elif x < lo:
            w = 0.5 * (1 - np.cos(np.pi * (x - (lo - t)) / t))
        elif x > hi:
            w = 0.5 * (1 + np.cos(np.pi * (x - hi) / t))
        else:
            w = 1.0
        return a.amp * w * 0.5 * (1 + np.sin(2 * np.pi * y / a.width))

    # bedrock (frozen, every node that has it)
    bk = find_name(s.frozen, "bedrock")
    bg, bv = s.frozen[bk]
    bv = np.array(bv, float)
    for i, g in enumerate(bg.tolist()):
        if g in Xd:
            bv[i] -= dz(Xd[g], Y[g])
    s.frozen[bk] = (bg, bv)
    bcol = {col(g): v for g, v in zip(bg.tolist(), bv.tolist()) if g in Xd}

    # the state: Zs (top nodes) and Zb (interface nodes), per column
    blocks = s.layout.split(s.x)
    sl = s.layout.slices()
    names = {b.name.lower(): b for b in s.layout.blocks}
    bzs, bzb = names["zs"], names["zb"]
    zs = np.array(blocks[bzs.name], float)
    zb = np.array(blocks[bzb.name], float)
    zsc = {col(g): i for i, g in enumerate(bzs.gids.tolist()) if g in Xd}
    zbc = {col(g): i for i, g in enumerate(bzb.gids.tolist()) if g in Xd}
    gmc = {col(g): i for i, g in enumerate(gg.tolist()) if g in Xd}
    nlow = nfloat = 0
    for c, ib in zbc.items():
        if c not in zsc or c not in gmc or c not in bcol:
            continue
        im, it = gmc[c], zsc[c]
        if gm[im] < -0.5:
            continue                                  # floating: unchanged
        d = dz(*c)
        if d <= 0.0:
            continue
        H = zs[it] - zb[ib]
        bed = bcol[c]
        if bed < 0.0 and H < -bed / r:                # thinner than flotation on the lowered bed: float it
            zb[ib] = -r * H
            zs[it] = (1.0 - r) * H
            gm[im] = -1.0
            nfloat += 1
        else:                                         # stays grounded: base on the bed, thickness kept
            zb[ib] = bed
            zs[it] = bed + H
            nlow += 1
    x = s.x.copy()
    x[sl[bzs.name]] = zs
    x[sl[bzb.name]] = zb
    s.frozen[gmk] = (gg, gm)
    # the grounding line after
    rows2 = {}
    for g, m in zip(gg.tolist(), gm.tolist()):
        if g in Xd:
            rows2.setdefault(round(Y[g], 1), []).append((Xd[g], m))
    gl2 = []
    for y, lst in sorted(rows2.items()):
        lst.sort()
        fl = [i for i, (_, m) in enumerate(lst) if m < -0.5]
        gl2.append(lst[fl[0] - 1][0] if fl and fl[0] > 0 else lst[-1][0])
    gl2 = np.array(gl2) / 1e3
    print(f"lowered by up to {a.amp:g} m: {nlow} grounded columns lowered, {nfloat} floated; grounding line now "
          f"{gl2.min():.2f} .. {gl2.max():.2f} km across the rows (waviness {gl2.max() - gl2.min():.2f} km)")
    meta = dict(s.meta)
    meta["perturb_bed"] = f"amp {a.amp} up {a.up} down {a.down} taper {a.taper} from {a.src}"
    save_state(a.out, s.layout, x, s.p, frozen=s.frozen, diag=s.diag, **meta)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
