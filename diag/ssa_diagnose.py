#!/usr/bin/env python3
"""ssa_diagnose.py

WHY does a (fully temperate) marine steady state not match the SSA steady state? Uses only the state's own
fields (Zs, Zb, bedrock, Velocity 1, GroundedMask) and the parameters of the run. Three questions, separately:

  A. Is it steady?            q(x) = int u dz  against  a x   (uniform accumulation, no basal melt)
  B. Is the local balance the SSA one, with the run's C and eta?
        d/dx(4 eta H u_m,x) - C u_b - rho_i g H s_x = 0
     C_eff = driving stress / u_b (+ the longitudinal term) is printed as a ratio to fcoeff: 1 = the sliding law
     and viscosity of the parameters file, anything else = the model's friction is not what is assumed.
  C. Does Schoof's condition hold at the grounding line?
        N = 4 eta H u_x  =  1/2 rho_i g (1 - rho_i/rho_w) H^2      (N / that, close to the GL, 1 = holds)
        H_g = flotation, and  a x_g = c K H_g^(5/2)  with c = 1.045 for the full SSA (1 = Schoof's asymptotics)

Needs the state to hold the Velocity 1 diagnostic (the marine configs write it).

Usage (repo root):
    python3 diag/ssa_diagnose.py work/flowline_rect_coarse/steady_temperate.npz
    options: --repo . --params case/parameters_fine2x.lua --smooth-km 5
Writes ssa_diagnose.png.
"""
import argparse
import math
import os
import re
import sys

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


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


def rms(v):
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v**2))) if len(v) else float("nan")


def smooth_uniform(x, v, dx, width):
    xu = np.arange(x[0], x[-1] + dx, dx)
    vu = np.interp(xu, x, v)
    n = max(int(round(width / dx)), 1)
    if n > 1:
        pad = np.pad(vu, (n // 2, n - 1 - n // 2), mode="edge")
        vu = np.convolve(pad, np.ones(n) / n, mode="valid")
    return xu, vu


def n_exact(x, s, zb, us, ub, a, eta, dx, width):
    """depth-integrated resistive stress N = 4 eta int u_x dz = 4 eta (q_x - u_s s_x + u_b zb_x), q_x = a (steady),
    exact for incompressible plane strain (sigma'_zz = -sigma'_xx); no plug-flow assumption"""
    X, S = smooth_uniform(x, s, dx, width)
    _, ZB = smooth_uniform(x, zb, dx, width)
    _, US = smooth_uniform(x, us, dx, width)
    _, UB = smooth_uniform(x, ub, dx, width)
    return X, 4 * eta * (a - US * np.gradient(S, X) + UB * np.gradient(ZB, X))


def load_columns(path, repo):
    sys.path.insert(0, repo)
    from ess.masks import _on
    from ess.statefile import find_name, load_state
    s = load_state(path)
    d = lambda n: s.diag[find_name(s.diag, n)]
    gx, X = d("coordinate 1")
    gz, Z = d("coordinate 3")
    try:
        gu, U = d("velocity 1")
    except KeyError:
        sys.exit("this state has no 'Velocity 1' diagnostic")
    parts = s.layout.split(np.asarray(s.x))
    blk = lambda name: next(b for b in s.layout.blocks if b.name.lower() == name)
    zsb, zbb = blk("zs"), blk("zb")
    key = lambda v: np.round(v, 1)

    def per_x(gids, vals):
        x = key(_on(gx, X, gids))
        ok = np.isfinite(x) & np.isfinite(vals)
        ux, inv = np.unique(x[ok], return_inverse=True)
        return dict(zip(ux, np.bincount(inv, weights=np.asarray(vals)[ok]) / np.bincount(inv)))

    zs = per_x(zsb.gids, parts[zsb.name])
    zb = per_x(zbb.gids, parts[zbb.name])
    bed = per_x(*s.frozen[find_name(s.frozen, "bedrock")]) if any(k.lower() == "bedrock" for k in s.frozen) else {}
    gm = per_x(*s.frozen[find_name(s.frozen, "GroundedMask")])
    ctm = s.frozen[find_name(s.frozen, "ColdTempMask")] if any(k.lower() == "coldtempmask" for k in s.frozen) else None
    temperate = None
    if ctm is not None:
        gmv = _on(*s.frozen[find_name(s.frozen, "GroundedMask")], ctm[0], fill=-1.0)
        if (gmv > -0.5).any():
            temperate = float(np.mean(np.asarray(ctm[1])[gmv > -0.5] > 0))

    xu_, zu_ = key(_on(gx, X, gu)), np.round(_on(gz, Z, gu), 2)
    cell = {}
    for xi, zi, ui in zip(xu_, zu_, U):
        if np.isfinite(xi) and np.isfinite(zi) and np.isfinite(ui):
            cell.setdefault(xi, {}).setdefault(zi, []).append(ui)
    rows = []
    for xi in sorted(set(zs) & set(zb) & set(cell) & set(gm)):
        pts = sorted((zi, float(np.mean(v))) for zi, v in cell[xi].items() if zb[xi] - 0.5 <= zi <= zs[xi] + 0.5)
        if len(pts) < 2:
            continue
        zz, uu = np.array(pts).T
        H = zs[xi] - zb[xi]
        if H < 1.0:
            continue
        q = _trapz(uu, zz) * H / max(zz[-1] - zz[0], 1e-9)
        rows.append((xi, H, zs[xi], zb[xi], bed.get(xi, np.nan), q / H, uu[0], uu[-1], q, gm[xi]))
    R = np.array(rows)
    return R, temperate, s.p


def load_columns_vtu(folder, tag):
    """The same per-column rows from an Elmer .vtu output (partitioned): ice nodes of a column are those from the base
    node (|groundedmask| = 1 there, 0 on every other node; zb is non-zero only there) up to the top node; rock nodes
    below have no velocity. Dedupes the nodes shared between partitions. The plane y = min(y) is used."""
    import glob
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vtu_read import read_many
    fs = sorted(glob.glob(os.path.join(folder, f"*np*_t{tag}*.vtu"))) or sorted(glob.glob(os.path.join(folder, f"*t{tag}*.vtu")))
    if not fs:
        sys.exit(f"no .vtu files for tag {tag!r} in {folder}")
    P, pt = read_many(fs)
    need = ("velocity", "groundedmask", "bedrock")
    for k in need:
        if k not in pt:
            sys.exit(f"the .vtu files have no point field {k!r}: {sorted(pt)}")
    on = np.isclose(P[:, 1], P[:, 1].min())
    P, gm_n, bed_n, vel = P[on], pt["groundedmask"][on], pt["bedrock"][on], pt["velocity"][on]
    ctm_n = pt["coldtempmask"][on] if "coldtempmask" in pt else np.zeros(len(P))
    T_n = pt["temperature"][on] if "temperature" in pt else np.zeros(len(P))
    _, keep = np.unique(np.round(P[:, [0, 2]], 2), axis=0, return_index=True)
    P, gm_n, bed_n, vel, ctm_n, T_n = P[keep], gm_n[keep], bed_n[keep], vel[keep], ctm_n[keep], T_n[keep]
    xk = np.round(P[:, 0], 1)
    rows, nfall, temp_g, p = [], 0, [], np.nan
    for xi in np.unique(xk):
        c = np.where(xk == xi)[0]
        c = c[np.argsort(P[c, 2])]
        z = P[c, 2]
        base = np.where(np.abs(gm_n[c]) > 0.5)[0]
        if len(base):
            ib = base[0]
        else:                                           # no marked base node: the lowest node that moves
            mv = np.where(np.abs(vel[c, 0]) > 0)[0]
            if not len(mv):
                continue
            ib, nfall = mv[0], nfall + 1
        zz, uu = z[ib:], vel[c, 0][ib:]
        H = zz[-1] - zz[0]
        if H < 1.0 or len(zz) < 2:
            continue
        q = _trapz(uu, zz)
        gmv = 1.0 if gm_n[c][ib] > -0.5 else -1.0
        rows.append((xi, H, zz[-1], zz[0], float(np.median(bed_n[c])), q / H, uu[0], uu[-1], q, gmv))
        if gmv > 0:
            temp_g.append(ctm_n[c][ib] > 0)
        if not np.isfinite(p):
            p = float(T_n[c][-1])
    if nfall:
        print(f"note: {nfall} columns had no node with |groundedmask| = 1; used the lowest moving node as the base")
    return np.array(rows), (float(np.mean(temp_g)) if temp_g else None), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", nargs="?", default=None)
    ap.add_argument("--vtu", help="folder with a partitioned Elmer .vtu output instead of an .npz state")
    ap.add_argument("--tag", default="", help="with --vtu: the time-step tag in the file names (e.g. 0369)")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--params", default=None)
    ap.add_argument("--smooth-km", type=float, default=5.0, help="moving average along x before differentiating")
    ap.add_argument("--gl-smooth-km", type=float, default=0.5, help="moving average for the grounding-line part (C)")
    ap.add_argument("--out", default="ssa_diagnose")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    pf = args.params or os.path.join(repo, "case", "parameters_fine2x.lua")
    if not os.path.exists(pf):
        pf = os.path.join(repo, "case", "parameters.lua")
    P = read_lua_params(pf)
    eta, rhoi, rhow, g = P["viscosity"], P["rhoi"], P["rhow"], P["gravity"]
    a, C = P["aDIM"], P["fcoeff"]
    r, rg = rhoi / rhow, rhoi * g
    K = math.sqrt(1 / (2 * eta) * rg**2 * (1 - r) / (4 * C))
    print("parameters:", pf, "\neta = %.5g, fcoeff = %.5g, a = %.5g, rho_i/rho_w = %.4f" % (eta, C, a, r))

    if args.vtu:
        R, temperate, p = load_columns_vtu(args.vtu, args.tag)
        args.state = f"{args.vtu} (t{args.tag}, .vtu)"
    elif args.state:
        R, temperate, p = load_columns(args.state, repo)
    else:
        sys.exit("give a state .npz or --vtu FOLDER --tag TAG")
    x, H, s, zb, bed, um, ub, us, q, gm = R.T
    grounded = gm > -0.5
    xgl = x[grounded].max()
    print("state: %s  p = %g   %d columns, grounding line (last grounded column) %.2f km" % (args.state, p, len(x), xgl / 1e3))
    if temperate is not None:
        print("grounded bed temperate fraction %.3f%s" % (temperate, "" if temperate > 0.999 else "   <-- NOT fully temperate"))

    # ---------------- A. steady?
    sel = grounded & (x > 0.05 * xgl)
    ratio = q[sel] / (a * x[sel])
    print("\nA. mass balance  q / (a x) over the grounded ice (5%% of x_gl onwards): median %.3f, min %.3f, max %.3f"
          % (np.median(ratio), ratio.min(), ratio.max()))
    near = grounded & (x > xgl - 5e3)
    print("   at the grounding line (last 5 km): q / (a x) = %.3f   (1 = steady)" % np.mean(q[near] / (a * x[near])))

    # ---------------- smoothed profiles on a uniform grid over the grounded ice
    xx = x[grounded]
    dx = 500.0
    W = args.smooth_km * 1e3
    X_, Hs = smooth_uniform(xx, H[grounded], dx, W)
    _, S = smooth_uniform(xx, s[grounded], dx, W)
    _, Um = smooth_uniform(xx, um[grounded], dx, W)
    _, Ub = smooth_uniform(xx, ub[grounded], dx, W)
    sx, ux = np.gradient(S, X_), np.gradient(Um, X_)
    N = 4 * eta * Hs * ux
    Nx = np.gradient(N, X_)
    D = -rg * Hs * sx                                           # driving stress
    inner = (X_ > 0.05 * xgl) & (X_ < xgl - 3 * W)

    # ---------------- B. local balance
    Ceff_b = (D + Nx) / Ub                                      # SSA: C u_b = D + dN/dx
    Ceff_noL = D / Ub                                           # neglecting the longitudinal term
    Ceff_m = (D + Nx) / Um
    print("\nB. local balance  d/dx(4 eta H u_x) - C u - rho g H s_x = 0   (grounded ice, away from both ends)")
    print("   C_eff / fcoeff with the basal velocity  u_b : median %.3f   (%.3f .. %.3f at the 10th/90th percentile)"
          % (np.median(Ceff_b[inner]) / C, np.percentile(Ceff_b[inner], 10) / C, np.percentile(Ceff_b[inner], 90) / C))
    print("   C_eff / fcoeff with the depth mean     u_m : median %.3f" % (np.median(Ceff_m[inner]) / C))
    print("   longitudinal term / driving stress          : %.3g" % (rms(Nx[inner]) / rms(D[inner])))
    print("   u_b / u_m  (1 = plug flow; < 1 = vertical shear): median %.3f" % np.median(Ub[inner] / Um[inner]))
    res = Nx - C * Ub + D
    print("   residual with the parameters' C and u_b / driving: %.3g   (0 = the model obeys the SSA balance)"
          % (rms(res[inner]) / rms(D[inner])))

    # ---------------- C. grounding-line condition (light smoothing only: the boundary layer is a few km wide)
    xg_, Hg_ = smooth_uniform(xx, H[grounded], 250.0, args.gl_smooth_km * 1e3)
    _, Ug_ = smooth_uniform(xx, um[grounded], 250.0, args.gl_smooth_km * 1e3)
    N2 = 4 * eta * Hg_ * np.gradient(Ug_, xg_)
    Nref2 = 0.5 * rg * (1 - r) * Hg_**2
    print("\nC. grounding-line condition  N / (1/2 rho g (1 - r) H^2)   (1 = Schoof's stress condition)")
    for dkm in (0.5, 1, 2, 5, 10, 20):
        j = np.argmin(np.abs(xg_ - (xgl - dkm * 1e3)))
        print("   %4.1f km upstream of the GL: %.3f" % (dkm, N2[j] / Nref2[j]))
    xe_, Ne_ = n_exact(xx, s[grounded], zb[grounded], us[grounded], ub[grounded], a, eta, 250.0, args.gl_smooth_km * 1e3)
    He_ = np.interp(xe_, xx, H[grounded])
    print("   the same with the EXACT depth-integrated stress N = 4 eta (a - u_s s_x + u_b zb_x)  (no plug-flow assumption):")
    for dkm in (0.25, 0.5, 1, 2, 5, 10, 20):
        j = np.argmin(np.abs(xe_ - (xgl - dkm * 1e3)))
        print("   %5.2f km upstream of the GL: %.3f" % (dkm, Ne_[j] / (0.5 * rg * (1 - r) * He_[j] ** 2)))
    xs_, Hs_, Us_ = x[grounded], H[grounded], um[grounded]
    o = np.argsort(xs_)
    xs_, Hs_, Us_ = xs_[o], Hs_[o], Us_[o]
    Nl = 4 * eta * Hs_[-1] * (Us_[-1] - Us_[-2]) / (xs_[-1] - xs_[-2])
    print("   last two grounded columns (one-sided difference, spacing %.0f m): N / ref = %.3f"
          % (xs_[-1] - xs_[-2], Nl / (0.5 * rg * (1 - r) * Hs_[-1] ** 2)))
    Hgl = float(Hs_[-1])
    zbg = np.interp(xgl, x[np.isfinite(bed)], bed[np.isfinite(bed)]) if np.isfinite(bed).any() else float("nan")
    hf = -zbg / r
    print("   H at the last grounded column %.1f m,  flotation thickness %.1f m  (ratio %.3f)" % (Hgl, hf, Hgl / hf))
    sel_b = grounded & (x > xgl - 4e3)
    if sel_b.sum() >= 3:
        sl_b = np.polyfit(x[sel_b], zb[sel_b], 1)[0]
        sel_s = grounded & (x > xgl - 4e3)
        sl_s = np.polyfit(x[sel_s], s[sel_s], 1)[0]
        print("   over the last 4 km of grounded ice: bed slope db/dx = %+.2e (+ = rises seaward), surface slope ds/dx = %+.2e, "
              "bed depth at the GL %.1f m" % (sl_b, sl_s, zb[grounded][np.argmax(x[grounded])]))
    print("   a x_gl / (K H_g^(5/2)) = %.3f   (full SSA: 1.045;  Schoof's asymptotics: 1)" % (a * xgl / (K * Hgl ** 2.5)))

    # ---------------- D. the floating shelf: a free 1D shelf has N = 1/2 rho g (1 - r) H^2 at EVERY x
    X3 = N3 = ref3 = None
    fl = (gm < -0.5) & (x > xgl)
    if fl.sum() > 5:
        o3 = np.argsort(x[fl])
        xf, Hf_, Uf_ = x[fl][o3], H[fl][o3], um[fl][o3]
        X3, H3 = smooth_uniform(xf, Hf_, 250.0, args.gl_smooth_km * 1e3)
        _, U3 = smooth_uniform(xf, Uf_, 250.0, args.gl_smooth_km * 1e3)
        N3 = 4 * eta * H3 * np.gradient(U3, X3)
        ref3 = 0.5 * rg * (1 - r) * H3**2
        print("\nD. floating shelf (%d columns, %.1f .. %.1f km):  N / (1/2 rho g (1 - r) H^2)   (a free shelf: 1 everywhere)"
              % (fl.sum(), xf[0] / 1e3, xf[-1] / 1e3))
        for dkm in (1, 2, 5, 10, 20, 40, 60):
            if xgl + dkm * 1e3 <= X3[-1]:
                j = np.argmin(np.abs(X3 - (xgl + dkm * 1e3)))
                print("   %3d km seaward of the GL: %.3f   (H %.0f m, u %.0f m/yr)" % (dkm, N3[j] / ref3[j], H3[j], U3[j]))
        xe3, Ne3 = n_exact(xf, s[fl][o3], zb[fl][o3], us[fl][o3], ub[fl][o3], a, eta, 250.0, args.gl_smooth_km * 1e3)
        He3 = np.interp(xe3, xf, Hf_)
        print("   EXACT depth-integrated stress on the shelf:")
        for dkm in (0.25, 0.5, 1, 2, 5, 20, 60):
            if xgl + dkm * 1e3 <= xe3[-1]:
                j = np.argmin(np.abs(xe3 - (xgl + dkm * 1e3)))
                print("   %5.2f km seaward of the GL: %.3f" % (dkm, Ne3[j] / (0.5 * rg * (1 - r) * He3[j] ** 2)))
        j = len(X3) - 3
        print("   near the front (x = %.1f km): %.3f   (H %.0f m, u %.0f m/yr)" % (X3[j] / 1e3, N3[j] / ref3[j], H3[j], U3[j]))
    else:
        print("\nD. no floating columns found in this state")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(x[grounded] / 1e3, q[grounded] / (a * x[grounded]), "k")
    ax[0, 0].axhline(1, color="gray", ls=":")
    ax[0, 0].set_ylabel("q / (a x)")
    ax[0, 0].set_title("A. steady? (1 = yes)")
    ax[0, 1].plot(X_ / 1e3, D, "k", label="driving  -rho g H s_x")
    ax[0, 1].plot(X_ / 1e3, C * Ub, "r", label="C u_b  (fcoeff)")
    ax[0, 1].plot(X_ / 1e3, D + Nx, "b--", label="driving + dN/dx")
    ax[0, 1].set_ylabel("MPa")
    ax[0, 1].legend()
    ax[0, 1].set_title("B. local balance")
    ax[1, 0].plot(X_ / 1e3, Ceff_b / C, "r", label="(D + dN/dx) / u_b")
    ax[1, 0].plot(X_ / 1e3, Ceff_noL / C, "b:", label="D / u_b")
    ax[1, 0].axhline(1, color="gray", ls=":")
    ax[1, 0].set_ylim(0, 3)
    ax[1, 0].set_ylabel("C_eff / fcoeff")
    ax[1, 0].set_xlabel("x (km)")
    ax[1, 0].legend()
    ax[1, 1].plot(xg_ / 1e3, N2 / Nref2, "k")
    ax[1, 1].axhline(1, color="gray", ls=":")
    ax[1, 1].set_ylim(0, 3)
    if X3 is not None:
        ax[1, 1].plot(X3 / 1e3, N3 / ref3, "b", label="shelf")
        ax[1, 1].legend()
    ax[1, 1].set_xlim(max(xgl / 1e3 - 60, 0), (X3[-1] / 1e3 + 2) if X3 is not None else xgl / 1e3 + 2)
    ax[1, 1].set_ylabel("N / (1/2 rho g (1-r) H^2)")
    ax[1, 1].set_xlabel("x (km)")
    ax[1, 1].set_title("C. Schoof's stress condition")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=130)
    print("\nplot:", args.out + ".png")


if __name__ == "__main__":
    main()
