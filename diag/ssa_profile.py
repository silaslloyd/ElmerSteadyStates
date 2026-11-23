#!/usr/bin/env python3
"""ssa_profile.py

Steady-state grounded ice-sheet profile, divide to grounding line, for a marine flowline with
EXACTLY the mechanical parameters of ElmerSteadyStates (case/parameters.lua):

  * linear (Newtonian) ice, constant viscosity eta;
  * linear Weertman sliding, tau_b = C u, C = fcoeff (fully temperate bed: exp(-Tr/gamma) = 1);
  * uniform accumulation aDIM, no basal melt  =>  steady flux q = aDIM x, depth-mean velocity u = q / H;
  * bed from case/marine_ics/bed_dim.xy; rho_i, rho_w, g from parameters.lua.

Plane-strain SSA force balance:   d/dx( 4 eta H u_x ) - C u - rho_i g H s_x = 0,   s = H + b.
Boundary conditions (the position x_g of the grounding line is an unknown, found by the solver):
  divide  x -> 0   : s_x = 0 (sliding balance  rho g H s_x = - C u  at the first point)
  GL      x = x_g  : H = h_f = -b(x_g) rho_w / rho_i                    (flotation)
                     4 eta H u_x = 1/2 rho_i g (1 - rho_i/rho_w) H^2    (Schoof's stress condition, unbuttressed shelf)
The boundary-layer result of Schoof (2007), a x_g = K h_f^(5/2), is printed for comparison.

Usage:
    python3 ssa_profile.py                           (from the repo root)
    python3 ssa_profile.py --repo ~/ElmerSteadyStates --state work/<run>/steady_p268.15.npz
Writes ssa_profile.png, ssa_profile.csv (x, H, s, b, u) and prints x_g, h_g, H(0).
"""
import argparse
import math
import os
import re
import sys

import numpy as np
from scipy.integrate import solve_bvp


def read_lua_params(path):
    env = {"math": math}
    for raw in open(path):
        line = raw.split("--")[0].strip().rstrip(";").strip()
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", line)
        if not m:
            continue
        name, expr = m.groups()
        try:
            env[name] = eval(expr.replace("^", "**"), {"__builtins__": {}}, env)
        except Exception:
            pass
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--params", default=None)
    ap.add_argument("--bed", default=None, help="bed table x(m) b(m); default case/marine_ics/bed_dim.xy")
    ap.add_argument("--guess", default=None, help="surface table for the first guess; default zs_dim.xy next to the bed")
    ap.add_argument("--friction", type=float, default=1.0, help="multiplier on fcoeff (default 1: the model's C)")
    ap.add_argument("--x0", type=float, default=1e-3, help="first point as a fraction of x_g")
    ap.add_argument("--state", default=None, help="optional marine steady-state npz: overlay its grounded surface")
    ap.add_argument("--out", default="ssa_profile")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    P = read_lua_params(args.params or os.path.join(repo, "case", "parameters.lua"))
    bedf = args.bed or os.path.join(repo, "case", "marine_ics", "bed_dim.xy")
    gsf = args.guess or os.path.join(os.path.dirname(bedf), "zs_dim.xy")
    zbf = os.path.join(os.path.dirname(bedf), "zb_dim.xy")

    eta, rhoi, rhow, g = P["viscosity"], P["rhoi"], P["rhow"], P["gravity"]
    a, C = P["aDIM"], args.friction * P["fcoeff"]
    r = rhoi / rhow
    rg = rhoi * g
    bt = np.loadtxt(bedf)
    bed = lambda x: np.interp(x, bt[:, 0], bt[:, 1])
    bed_x = lambda x: np.interp(x, bt[:, 0], np.gradient(bt[:, 1], bt[:, 0]))

    print("eta = %.6g MPa yr, C = %.6g MPa yr/m, a = %.6g m/yr, rho_i = %.6g, rho_w = %.6g (MPa m^-2 yr^2)"
          % (eta, C, a, rhoi, rhow))

    # --- Schoof (2007) boundary-layer flux condition, n = m = 1
    K = math.sqrt(1 / (2 * eta) * rg**2 * (1 - r) / (4 * C))       # A = 1 / (2 eta)
    xs = np.linspace(1e3, bt[-1, 0], 200001)
    f = K * (np.maximum(-bed(xs), 0) / r) ** 2.5 - a * xs
    i = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
    x_schoof = float(xs[i[0]]) if len(i) else float("nan")
    print("Schoof flux condition a x_g = K h_f^(5/2):  x_g = %.2f km, h_g = %.1f m" % (
        x_schoof / 1e3, -bed(x_schoof) / r if len(i) else float("nan")))

    # --- first guess from the Marine_Term tables
    gs = np.loadtxt(gsf)
    gb = np.loadtxt(zbf)
    xg0 = x_schoof if np.isfinite(x_schoof) else 686e3
    HS = 1000.0                                                    # m, scale of H

    NS = 0.5 * rg * (1 - r) * HS**2                                # scale of the longitudinal force N = 4 eta H u_x

    def rhs(xi, y, p):
        xg = p[0]
        x = xi * xg
        H = y[0] * HS
        N = y[1] * NS
        Hx = H * (a - N / (4 * eta)) / (a * x)                     # from H u_x = a - u H_x
        Nx = C * a * x / H + rg * H * (Hx + bed_x(x))              # force balance
        return np.vstack([xg * Hx / HS, xg * Nx / NS])

    x0 = args.x0

    def bc(ya, yb, p):
        xg = p[0]
        xa = x0 * xg
        Ha, Na = ya[0] * HS, ya[1] * NS
        ua = a * xa / Ha
        Hxa = -bed_x(xa) - C * ua / (rg * Ha)                      # sliding balance (s_x = -C u / rho g H) at the divide
        hf = -bed(xg) / r
        return np.array([
            (Na - 4 * eta * (a - ua * Hxa)) / NS,
            (yb[0] * HS - hf) / HS,                                # flotation
            (yb[1] * NS - 0.5 * rg * (1 - r) * hf**2) / NS,        # Schoof: 4 eta H u_x = 1/2 rho g (1 - r) H^2
        ])

    xi = np.linspace(x0, 1.0, 400)
    xx = xi * xg0
    H0 = np.maximum(np.interp(xx, gs[:, 0], gs[:, 1]) - np.interp(xx, gb[:, 0], gb[:, 1]), 50.0)
    H0[-1] = -bed(xg0) / r
    u0 = a * xx / H0
    N0 = 4 * eta * (a - u0 * np.gradient(H0, xx))
    N0[-1] = 0.5 * rg * (1 - r) * H0[-1] ** 2
    y0 = np.vstack([H0 / HS, N0 / NS])
    sol = solve_bvp(rhs, bc, xi, y0, p=[xg0], tol=1e-6, max_nodes=200000)
    print("solve_bvp:", sol.message)
    if not sol.success:
        sys.exit("no converged profile (try --x0 1e-2 or another --guess)")

    xg = float(sol.p[0])
    xf = np.linspace(x0, 1.0, 4000)
    Y = sol.sol(xf)
    x = xf * xg
    H = Y[0] * HS
    b = bed(x)
    s = H + b
    u = a * x / H
    print("\nSSA steady state with the model's mechanical parameters:")
    print("  grounding line x_g = %.2f km   (Schoof flux condition: %.2f km)" % (xg / 1e3, x_schoof / 1e3))
    print("  thickness at the GL = %.1f m (flotation %.1f m)   surface %.1f m" % (H[-1], -bed(xg) / r, s[-1]))
    print("  divide: H = %.1f m, surface = %.1f m" % (H[0], s[0]))
    print("  GL velocity u_g = %.1f m/yr, flux q_g = %.4g m^2/yr" % (u[-1], a * xg))

    np.savetxt(args.out + ".csv", np.c_[x, H, s, b, u], delimiter=",", header="x_m,H_m,s_m,bed_m,u_m_per_yr", comments="")
    print("table:", args.out + ".csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(x / 1e3, s, "r", label="SSA steady surface (this script)")
    ax[0].plot(x / 1e3, b, "k", label="bed")
    ax[0].plot(gs[:, 0] / 1e3, gs[:, 1], "g:", label="Marine_Term initial surface (zs_dim.xy)")
    if args.state:
        sys.path.insert(0, repo)
        from ess.masks import _get, _on, _xy_key
        from ess.statefile import load_state
        sv = load_state(args.state)
        zsb = next(bk for bk in sv.layout.blocks if bk.name.lower() == "zs")
        zv = sv.layout.split(np.asarray(sv.x))[zsb.name]
        xr = _on(*_get(sv.diag, "coordinate 1"), zsb.gids)
        o = np.argsort(xr)
        ax[0].plot(xr[o] / 1e3, zv[o], "b.", ms=3, label="model state Zs")
    ax[0].axvline(xg / 1e3, color="gray", ls="--")
    ax[0].set_ylabel("elevation (m)")
    ax[0].set_ylim(min(b.min(), -400), s.max() * 1.05)
    ax[0].legend()
    ax[1].plot(x / 1e3, u, "r")
    ax[1].set_ylabel("u = a x / H (m/yr)")
    ax[1].set_xlabel("x (km)")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=130)
    print("plot:", args.out + ".png")


if __name__ == "__main__":
    main()
