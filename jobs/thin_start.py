#!/usr/bin/env python3
"""A thin-ice start for growing a sheet (with sea level lowered: ess run --sea-level, jobs SEA_LEVEL), from any state on
the same mesh (it gives the layout, the bed, the columns):

    Zs = bed + H0, Zb = bed (every node grounded: sea level below the bed), GroundedMask 1, ColdTempMask -1 (cold),
    Water Sheet Thickness 1e-4 m, the ice temperature the column's surface value of the source (its top level; the
    surface condition sets it at the first step anyway), the bedrock on the steady geothermal profile below that
    (--keep-bedrock-T: the source's bedrock temperatures, with its streams' thermal pattern), velocity and pressure 0.

    python3 jobs/thin_start.py SRC.npz OUT.npz [--h0 100] [--sea-level -500]

--sea-level: the sea level the run starts at (m), only checked: the bed must be above it everywhere (land), else the
lowest bed and how many nodes are below are printed and nothing is written.

--float: instead, the ice floats where H0 is below flotation at --sea-level (default 0 then): H0 < (sea level - bed)
rho_w / rho_i (rho_i 920, rho_w 1003.27, as case/parameters.lua): base sea level - (rho_i/rho_w) H0, surface the rest
above, GroundedMask -1, temperate, sheet 0.1 m (hr); grounded elsewhere as above. E.g. --h0 300 --float: grounded where
the bed is above about -275 m, floating past the bend (a run at sea level 0, no SEA_LEVEL needed).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import find_name, load_state, save_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--h0", type=float, default=100.0, help="the initial ice thickness (m)")
    ap.add_argument("--sea-level", type=float, default=None)
    ap.add_argument("--float", action="store_true", help="float the ice where it is below flotation")
    ap.add_argument("--keep-bedrock-T", action="store_true",
                    help="keep the source's bedrock temperatures (by default they are reset: the column's surface "
                         "temperature at the bed, rising downwards with the steady geothermal gradient gthf / "
                         "heatconductivityrock of case/parameters.lua, so no thermal pattern of the source's ice "
                         "streams is left in the bed)")
    a = ap.parse_args()
    if a.sea_level is None:
        a.sea_level = 0.0 if a.float else -500.0
    r = 920.0 / 1003.27
    # the steady bedrock gradient (K/m) of case/parameters.lua: gthf / heatconductivityrock
    ys, M = 365.25 * 86400.0, 1e6
    rhoi, cp, k, grav, mu, Tsc = 920 / M / ys ** 2, 2097 * ys ** 2, 2.1 / M * ys, 9.81 * ys ** 2, 1e14 / M / ys, 5.0
    u = (k * Tsc / mu) ** 0.5
    xs = rhoi * cp ** 3 * u ** 5 * mu ** 2 / (k ** 3 * grav ** 2)
    zs = (mu * u * xs / (rhoi * grav)) ** (1 / 3)
    rock_grad = 0.5 * k * Tsc / zs / (3.0851 / M * ys)
    s = load_state(a.src)
    allf = {**s.diag, **s.frozen}
    gb, bed = s.frozen[find_name(s.frozen, "bedrock")]
    if not a.float and bed.min() <= a.sea_level:
        raise SystemExit(f"the bed reaches {bed.min():.1f} m, below the sea level {a.sea_level:g} m "
                         f"({int(np.sum(bed <= a.sea_level))} nodes): lower --sea-level")
    def on(rec, gids):                                  # a record's values at gids (sorted record ids)
        g, v = rec
        o = np.argsort(g)
        j = np.clip(np.searchsorted(g[o], gids), 0, len(g) - 1)
        if not np.array_equal(g[o][j], gids):
            raise SystemExit("a record does not cover the nodes it is needed at")
        return v[o][j]
    brec = (gb, bed)
    nb = int(np.min([len(b.gids) for b in s.layout.blocks if b.name.lower() in ("zs", "zb")]))
    x = s.x.copy()
    sl = s.layout.slices()
    blocks = s.layout.split(s.x)
    for b in s.layout.blocks:
        nm = b.name.lower()
        if nm == "zs":
            bb = on(brec, b.gids)
            fl = a.float & (r * a.h0 < a.sea_level - bb)
            x[sl[b.name]] = np.where(fl, a.sea_level + (1 - r) * a.h0, bb + a.h0)
        elif nm == "zb":
            bb = on(brec, b.gids)
            fl = a.float & (r * a.h0 < a.sea_level - bb)
            x[sl[b.name]] = np.where(fl, a.sea_level - r * a.h0, bb)
        elif nm == "temperature":
            T = blocks[b.name].copy()
            g = b.gids
            z = on(allf[find_name(allf, "coordinate 3")], g)
            bz = on(brec, g)
            ice = z >= bz - 1e-3
            # the column's surface temperature: the top level (largest gid of the column, nb columns)
            col = (g - 1) % nb
            top = np.full(nb, np.nan)
            lev = (g - 1) // nb
            o = np.argsort(lev)
            top[col[o]] = T[o]                          # the last write per column is its top level
            T[ice] = top[col[ice]]
            if a.keep_bedrock_T:
                msg = f"{int((~ice).sum())} bedrock nodes kept"
            else:                                       # the old run's bed (warm under its streams) would be kept
                T[~ice] = top[col[~ice]] + rock_grad * (bz[~ice] - z[~ice])
                msg = (f"{int((~ice).sum())} bedrock nodes on the steady geothermal profile from it "
                       f"({rock_grad * 1e3:.3f} K/km, {T[~ice].min():.2f} .. {T[~ice].max():.2f} K)")
            x[sl[b.name]] = T
            print(f"temperature: {int(ice.sum())} ice nodes set to their column's surface value "
                  f"({np.nanmin(top):.2f} .. {np.nanmax(top):.2f} K), {msg}")
    frozen = dict(s.frozen)
    nfl = 0
    for nm, val, vfl in (("GroundedMask", 1.0, -1.0), ("ColdTempMask", -1.0, 1.0), ("Water Sheet Thickness", 1e-4, 0.1)):
        try:
            k = find_name(frozen, nm)
        except KeyError:
            continue
        g, _ = frozen[k]
        fl = a.float & (r * a.h0 < a.sea_level - on(brec, g))
        frozen[k] = (g, np.where(fl, vfl, val))
        nfl = int(fl.sum())
    for k in [k for k in frozen if k.lower() == "remesh distance"]:
        del frozen[k]
    diag = {}
    for k, (g, v) in s.diag.items():
        kl = k.lower()
        if kl.startswith("coordinate"):
            diag[k] = (g, v)
        elif kl in ("velocity 1", "velocity 2", "velocity 3", "pressure"):
            diag[k] = (g, np.zeros(len(g)))
    save_state(a.out, s.layout, x, s.p, frozen, diag, source=a.src,
               thin_start=f"H0 {a.h0:g} m, sea level at the start {a.sea_level:g} m")
    if a.float:
        print(f"wrote {a.out}: H = {a.h0:g} m everywhere, sea level {a.sea_level:g} m: floating where the bed is below "
              f"{a.sea_level - r * a.h0:.1f} m ({nfl} base nodes), grounded and cold elsewhere")
    else:
        print(f"wrote {a.out}: H = {a.h0:g} m everywhere on the bed ({bed.min():.1f} .. {bed.max():.1f} m), all "
              f"grounded and cold; run with SEA_LEVEL starting at {a.sea_level:g}")


if __name__ == "__main__":
    main()
