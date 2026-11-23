#!/usr/bin/env python3
"""ssa_profile.py

Steady-state grounded ice-sheet profile, divide to grounding line, for a marine flowline with
EXACTLY the mechanical parameters of ElmerSteadyStates (case/parameters*.lua):

  * linear (Newtonian) ice, constant viscosity eta;
  * linear Weertman sliding, tau_b = C u, C = fcoeff (fully temperate bed: exp(-Tr/gamma) = 1);
  * uniform accumulation aDIM, no basal melt  =>  steady flux q = aDIM x, depth-mean velocity u = q / H;
  * rho_i, rho_w, g from the parameters file.

Plane-strain SSA force balance:   d/dx( 4 eta H u_x ) - C u - rho_i g H s_x = 0,   s = H + b.
Boundary conditions (the position x_g of the grounding line is an unknown, found by the solver):
  divide  x -> 0   : s_x = 0 (sliding balance  rho g H s_x = - C u  at the first point)
  GL      x = x_g  : H = h_f = -b(x_g) rho_w / rho_i                    (flotation)
                     4 eta H u_x = 1/2 rho_i g (1 - rho_i/rho_w) H^2    (Schoof's stress condition, unbuttressed shelf)
The boundary-layer result of Schoof (2007), a x_g = K h_f^(5/2), is printed for comparison.

The bed:
  --state STATE.npz : the bed of that marine state (frozen record 'bedrock', the base level of each column), and the
                      solver starts from the state's own surface. Prints the model's grounding line and the misfit
                      of its surface to the SSA surface, and overlays Zs. Use the fully temperate steady state
                      (e.g. work/flowline_rect_coarse/steady_temperate.npz, or the first point of the sweep).
  no state          : the Marine_Term table case/marine_ics/bed_dim.xy.

Usage (repo root):
    python3 diag/ssa_profile.py --state work/flowline_rect_coarse/steady_temperate.npz
    python3 diag/ssa_profile.py                       (table bed)
Writes ssa_profile.png and ssa_profile.csv (x, H, s, b, u).
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


def smooth(x, v, dx, width):
    """v(x) on a uniform grid of spacing dx, moving average over `width` metres"""
    xu = np.arange(x[0], x[-1] + dx, dx)
    vu = np.interp(xu, x, v)
    n = max(int(round(width / dx)), 1)
    if n > 1:
        k = np.ones(n) / n
        pad = np.pad(vu, (n // 2, n - 1 - n // 2), mode="edge")
        vu = np.convolve(pad, k, mode="valid")
    return xu, vu


def per_x(x, v, digits=1):
    """mean of v over the nodes sharing x (the nodes across the flowline), sorted by x"""
    ok = np.isfinite(x) & np.isfinite(v)
    ux, inv = np.unique(np.round(x[ok], digits), return_inverse=True)
    n = np.bincount(inv).astype(float)
    return ux, np.bincount(inv, weights=v[ok]) / n


def read_state(path, repo):
    sys.path.insert(0, repo)
    from ess.masks import _get, _on
    from ess.statefile import find_name, load_state
    s = load_state(path)
    gx, X = s.diag[find_name(s.diag, "coordinate 1")]
    zsb = next(b for b in s.layout.blocks if b.name.lower() == "zs")
    zs = s.layout.split(np.asarray(s.x))[zsb.name]
    xs = _on(gx, X, zsb.gids)
    out = {"zs": per_x(xs, zs)}
    if any(k.lower() == "bedrock" for k in s.frozen):
        g, v = s.frozen[find_name(s.frozen, "bedrock")]
        out["bed"] = per_x(_on(gx, X, g), np.asarray(v, float))
    else:
        raise SystemExit("this state has no frozen 'bedrock' record: use the table bed (no --state)")
    gm = _get(s.frozen, "GroundedMask")
    ct = _get(s.frozen, "ColdTempMask")
    out["gl"] = None
    try:
        from ess.spectrum import gl_position
        out["gl"] = float(gl_position(s.frozen, s.diag))
    except Exception:
        if gm is not None:
            xg_ = _on(gx, X, gm[0])
            out["gl"] = float(np.nanmax(xg_[np.asarray(gm[1]) > -0.5]))
    out["temperate"] = None
    if ct is not None and gm is not None:
        gr = _on(*gm, ct[0], fill=-1.0) > -0.5
        if gr.any():
            out["temperate"] = float(np.mean(np.asarray(ct[1])[gr] > 0))
    out["p"] = float(s.p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--params", default=None)
    ap.add_argument("--state", default=None, help="marine state npz: its bed and surface are used")
    ap.add_argument("--bed", default=None, help="bed table x(m) b(m); default case/marine_ics/bed_dim.xy")
    ap.add_argument("--bed-smooth-km", type=float, default=0.5, help="moving average of the bed (its slope enters the balance); other widths are tried if the solver fails")
    ap.add_argument("--friction", type=float, default=1.0, help="multiplier on fcoeff (default 1: the model's C)")
    ap.add_argument("--x0", type=float, default=1e-3, help="first point as a fraction of x_g")
    ap.add_argument("--out", default="ssa_profile")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    pfile = args.params
    if not pfile:
        pfile = os.path.join(repo, "case", "parameters_fine2x.lua")
        if not (args.state and os.path.exists(pfile)):
            pfile = os.path.join(repo, "case", "parameters.lua")
    P = read_lua_params(pfile)
    eta, rhoi, rhow, g = P["viscosity"], P["rhoi"], P["rhow"], P["gravity"]
    a, C = P["aDIM"], args.friction * P["fcoeff"]
    r = rhoi / rhow
    rg = rhoi * g
    print("parameters:", pfile)
    print("eta = %.6g MPa yr, C = %.6g MPa yr/m, a = %.6g m/yr, rho_i = %.6g, rho_w = %.6g (MPa m^-2 yr^2)"
          % (eta, C, a, rhoi, rhow))

    model = None
    if args.state:
        model = read_state(args.state, repo)
        xb, bb = model["bed"]
        print("state: %s  p = %g" % (args.state, model["p"]))
        print("model grounding line: %s km" % ("%.2f" % (model["gl"] / 1e3) if model["gl"] else "n/a"))
        if model["temperate"] is not None:
            print("grounded bed temperate fraction: %.3f%s" % (
                model["temperate"], "" if model["temperate"] > 0.999 else "   <-- NOT fully temperate: C = fcoeff is wrong here"))
    else:
        bt = np.loadtxt(args.bed or os.path.join(repo, "case", "marine_ics", "bed_dim.xy"))
        xb, bb = bt[:, 0], bt[:, 1]
    def attempt(width, x0):
        xbs, bbs = smooth(xb, bb, 500.0, width * 1e3)
        bed = lambda x: np.interp(x, xbs, bbs)
        bed_x = lambda x: np.interp(x, xbs, np.gradient(bbs, xbs))

        # --- Schoof (2007) boundary-layer flux condition, n = m = 1
        K = math.sqrt(1 / (2 * eta) * rg**2 * (1 - r) / (4 * C))       # A = 1 / (2 eta)
        xs = np.linspace(1e3, xbs[-1], 200001)
        f = K * (np.maximum(-bed(xs), 0) / r) ** 2.5 - a * xs
        i = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
        x_schoof = float(xs[i[0]]) if len(i) else float("nan")

        # --- first guess
        if model is not None and model["gl"]:
            xg0 = model["gl"]
        else:
            xg0 = x_schoof if np.isfinite(x_schoof) else 686e3
        xi = np.linspace(x0, 1.0, 400)
        xx = xi * xg0
        if model is not None:
            H0 = np.interp(xx, *model["zs"]) - bed(xx)
        else:
            zt = os.path.join(repo, "case", "marine_ics")
            gs, gb = np.loadtxt(os.path.join(zt, "zs_dim.xy")), np.loadtxt(os.path.join(zt, "zb_dim.xy"))
            H0 = np.interp(xx, gs[:, 0], gs[:, 1]) - np.interp(xx, gb[:, 0], gb[:, 1])
        H0 = np.maximum(H0, 50.0)
        H0[-1] = -bed(xg0) / r
        HS = 1000.0
        NS = 0.5 * rg * (1 - r) * HS**2                                # scale of the longitudinal force N = 4 eta H u_x

        def rhs(xi_, y, p):
            xg = p[0]
            x = xi_ * xg
            H = y[0] * HS
            N = y[1] * NS
            Hx = H * (a - N / (4 * eta)) / (a * x)                     # from H u_x = a - u H_x
            Nx = C * a * x / H + rg * H * (Hx + bed_x(x))              # force balance
            return np.vstack([xg * Hx / HS, xg * Nx / NS])

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

        u0 = a * xx / H0
        N0 = 4 * eta * (a - u0 * np.gradient(H0, xx))
        N0[-1] = 0.5 * rg * (1 - r) * H0[-1] ** 2
        y0 = np.vstack([H0 / HS, N0 / NS])
        sol = solve_bvp(rhs, bc, xi, y0, p=[xg0], tol=1e-6, max_nodes=200000)
        return sol, bed, xbs, bbs, x_schoof, HS

    sol = None
    for width, x0 in [(args.bed_smooth_km, args.x0), (0.25, args.x0), (1.0, args.x0), (args.bed_smooth_km, 1e-2),
                      (3.0, 1e-2), (0.0, 1e-2)]:
        sol, bed, xbs, bbs, x_schoof, HS = attempt(width, x0)
        print("solve_bvp (bed smoothing %g km, first point %g x_g): %s" % (width, x0, sol.message))
        if sol.success:
            break
    if not sol.success:
        sys.exit("no converged profile")
    print("Schoof flux condition a x_g = K h_f^(5/2):  x_g = %.2f km, h_g = %.1f m" % (
        x_schoof / 1e3, -bed(x_schoof) / r if np.isfinite(x_schoof) else float("nan")))

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
    if model is not None:
        zm = np.interp(x, *model["zs"])
        d = zm - s
        inner = (x > 0.02 * xg) & (x < 0.95 * xg)
        print("\nmodel vs SSA (grounded ice, 2%..95% of x_g):")
        if model["gl"]:
            print("  grounding line: model %.2f km, SSA %.2f km  (difference %.2f km)"
                  % (model["gl"] / 1e3, xg / 1e3, (model["gl"] - xg) / 1e3))
        print("  surface Zs_model - s_SSA: mean %.1f m, rms %.1f m, max |.| %.1f m  (SSA relief %.0f m)"
              % (d[inner].mean(), np.sqrt(np.mean(d[inner] ** 2)), np.abs(d[inner]).max(), s.max() - s.min()))
        print("  divide thickness: model %.1f m, SSA %.1f m"
              % (np.interp(0.0, *model["zs"]) - bed(0.0), H[0]))

    np.savetxt(args.out + ".csv", np.c_[x, H, s, b, u], delimiter=",", header="x_m,H_m,s_m,bed_m,u_m_per_yr", comments="")
    print("table:", args.out + ".csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(x / 1e3, s, "r", label="SSA steady surface (this script)")
    ax[0].plot(xbs[xbs <= x[-1] * 1.05] / 1e3, bbs[xbs <= x[-1] * 1.05], "k", label="bed")
    if model is not None:
        mz = model["zs"]
        keep = mz[0] <= x[-1] * 1.05
        ax[0].plot(mz[0][keep] / 1e3, mz[1][keep], "b.", ms=3, label="model state Zs")
        if model["gl"]:
            ax[0].axvline(model["gl"] / 1e3, color="b", ls=":", label="model grounding line")
    ax[0].axvline(xg / 1e3, color="r", ls="--", label="SSA grounding line")
    ax[0].set_ylabel("elevation (m)")
    ax[0].legend()
    ax[1].plot(x / 1e3, u, "r")
    ax[1].set_ylabel("u = a x / H (m/yr)")
    ax[1].set_xlabel("x (km)")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=130)
    print("plot:", args.out + ".png")


if __name__ == "__main__":
    main()
