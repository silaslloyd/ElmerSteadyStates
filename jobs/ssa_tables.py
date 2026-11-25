#!/usr/bin/env python3
"""The Marine_Term initial-state tables (case/marine_ics: zs_dim.xy, zb_dim.xy, bed_dim.xy, Tb_dim.xy, ctmask_dim.xy) for
the STEADY SSA PROFILE of diag/ssa_profile.py instead of the Schoof one (686 km): the model's own mechanical
parameters, the grounding line where the SSA solution puts it (585.57 km on the Marine_Term bed), grounded ice from the
profile, the shelf from the unbuttressed linear-viscosity shelf equations:

    4 eta H u_x = 1/2 rho_i g (1 - rho_i/rho_w) H^2     (stress condition, holds all along the shelf)
    H u = a x                                           (steady flux, uniform accumulation a, no melt)
    =>  H' = H (a - kappa H^2) / (a x),  kappa = rho_i g (1 - rho_i/rho_w) / (8 eta),   from H(x_g) = the profile's.

Floating ice: Zs = H (1 - rho_i/rho_w), Zb = Zs - H. Grounded: Zb = the bed table, Zs from the profile. The other tables
(bed, base temperature, cold/temperate mask) are copied unchanged. `ess init` with [case] marine_ics = this directory then
starts from the SSA steady state; ess/case.py takes zb = max(zb, bed) and the grounded mask from zb - bed.

    python3 diag/ssa_profile.py --params case/parameters_fine2x.lua --out work/ssa_start/ssa_profile
    python3 jobs/ssa_tables.py work/ssa_start/ssa_profile.csv case/marine_ics_ssa
(jobs/ssa_start.sh does both and the rest.)  --params default case/parameters_fine2x.lua, --ics default case/marine_ics."""
import argparse
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "diag"))
from ssa_profile import read_lua_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="ssa_profile.py's table (x_m, H_m, s_m, bed_m, u)")
    ap.add_argument("out", help="the new tables' directory")
    ap.add_argument("--params", default=os.path.join(ROOT, "case", "parameters_fine2x.lua"))
    ap.add_argument("--ics", default=os.path.join(ROOT, "case", "marine_ics"), help="the tables to take the grid and the rest from")
    a = ap.parse_args()

    P = read_lua_params(a.params)
    eta, rhoi, rhow, g, acc = P["viscosity"], P["rhoi"], P["rhow"], P["gravity"], P["aDIM"]
    r = rhoi / rhow
    kappa = rhoi * g * (1 - r) / (8 * eta)

    prof = np.loadtxt(a.csv, delimiter=",", skiprows=1)
    xp, Hp, sp = prof[:, 0], prof[:, 1], prof[:, 2]
    xg, Hg = xp[-1], Hp[-1]
    grid = np.loadtxt(os.path.join(a.ics, "zs_dim.xy"))[:, 0]
    bed_t = np.loadtxt(os.path.join(a.ics, "bed_dim.xy"))
    bed = np.interp(grid, bed_t[:, 0], bed_t[:, 1])

    # the shelf: H' = H (a - kappa H^2) / (a x) from the grounding line (RK4 on the table's grid)
    sh = grid > xg
    xs = np.concatenate([[xg], grid[sh]])
    Hs = np.empty(len(xs))
    Hs[0] = Hg
    f = lambda x, H: H * (acc - kappa * H * H) / (acc * x)
    for i in range(len(xs) - 1):
        h = xs[i + 1] - xs[i]
        k1 = f(xs[i], Hs[i]); k2 = f(xs[i] + h / 2, Hs[i] + h / 2 * k1)
        k3 = f(xs[i] + h / 2, Hs[i] + h / 2 * k2); k4 = f(xs[i] + h, Hs[i] + h * k3)
        Hs[i + 1] = Hs[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    zs = np.empty(len(grid))
    zb = np.empty(len(grid))
    gr = grid <= xg
    zs[gr] = np.interp(grid[gr], xp, sp, left=sp[0])        # the divide: the first profile point's surface (s_x = 0 there)
    zb[gr] = bed[gr]
    Hf = Hs[1:]
    zs[sh] = Hf * (1 - r)
    zb[sh] = zs[sh] - Hf
    # the last grounded grid point to the first floating one: the profile reaches the flotation thickness at xg
    os.makedirs(a.out, exist_ok=True)
    for name in os.listdir(a.ics):
        if name.endswith(".xy") and name not in ("zs_dim.xy", "zb_dim.xy"):
            shutil.copy(os.path.join(a.ics, name), os.path.join(a.out, name))
    np.savetxt(os.path.join(a.out, "zs_dim.xy"), np.c_[grid, zs], fmt="%.8g")
    np.savetxt(os.path.join(a.out, "zb_dim.xy"), np.c_[grid, zb], fmt="%.8g")
    print(f"ssa_tables: grounding line {xg / 1e3:.2f} km, H_g {Hg:.1f} m; shelf thickness {Hs[1]:.1f} m at the first shelf point, "
          f"{Hs[-1]:.1f} m at {xs[-1] / 1e3:.0f} km; divide Zs {zs[0]:.1f} m  -> {a.out}")


if __name__ == "__main__":
    main()
