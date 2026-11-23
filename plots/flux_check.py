#!/usr/bin/env python3
"""Mass balance of a (supposedly) steady flowline: the ice flux q(x) = integral of u dz over each column (Zb to Zs)
against the accumulation integrated from the divide, a x (uniform surface accumulation a, no basal mass balance:
Zb Accumulation 0 in both ElmerStreams' Flowline and ours). In a steady state q(x) = a x everywhere, grounded and
floating, so q / (a x) = 1; at the grounding line in particular q_gl = a x_gl.

Input: an ESS state (.npz with diagnostics Velocity 1 and the coordinates) or an Elmer VTU output directory / file
(its last time step, as plots/vtu_profile.py: point fields Velocity (3 components) or Velocity 1, Zb, bedrock).
Columns: nodes sharing (x, y); per column the ice nodes (z >= the column's Zb - 0.5 m), u integrated by the
trapezoid rule in z; averaged across y at each x.

    python3 plots/flux_check.py STATE.npz | VTU_DIR | F.pvtu  [--a 0.0321394] [--every-km 50]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
A_DIM = 0.03213938607148378            # aDIM of case/parameters_fine2x.lua = ElmerStreams' (m/yr)


def from_state(path):
    from ess.statefile import find_name, load_state
    s = load_state(path)
    look = lambda d, n: d[find_name(d, n)]
    gx, X = look(s.diag, "coordinate 1")
    gy, Y = look(s.diag, "coordinate 2")
    gz, Z = look(s.diag, "coordinate 3")
    gu, U = look(s.diag, "velocity 1")
    o = np.argsort(gx)
    pos = lambda g: o[np.searchsorted(gx[o], g)]
    if not (np.array_equal(gx, gy) and np.array_equal(gx, gz)):
        raise SystemExit("coordinate records with different node sets")
    i = pos(gu)
    x, y, z, u = X[i], Y[i], Z[i], U
    # Zb per column from the state's Zb block (on the base nodes): looked up by (x, y)
    sl = s.layout.slices()
    b = next(b_ for b_ in s.layout.blocks if b_.name.lower() == "zb")
    j = pos(b.gids)
    zbx, zby, zbv = X[j], Y[j], s.x[sl[b.name]]
    gl = None
    try:
        from ess.spectrum import gl_position
        gl = gl_position(s.frozen, s.diag)
    except Exception:
        pass
    return x, y, z, u, (zbx, zby, zbv), gl, path


def from_vtu(arg):
    from vtu_profile import field, pick, pieces_of, read_vtu
    files, what = (pieces_of(arg), arg) if os.path.isfile(arg) else pick(arg)
    P, F = [], {}
    for p in files:
        r = read_vtu(p)
        P.append(r["points"])
        for k, v in r["fields"].items():
            F.setdefault(k, []).append(v)
    P = np.concatenate(P)
    F = {k: np.concatenate(v) for k, v in F.items() if sum(len(x) for x in v) == len(P)}
    kv = field(F, "velocity")
    if kv is not None and F[kv].ndim == 2:
        u = F[kv][:, 0]
    else:
        kv = field(F, "velocity 1", "velocity_1")
        if kv is None:
            raise SystemExit(f"no velocity field in {what}: {sorted(F)}")
        u = F[kv]
    kb = field(F, "zb")
    if kb is None:
        raise SystemExit("no Zb field")
    m = F[kb] != 0                               # Elmer writes 0 where Zb is not defined
    zb = (P[m, 0], P[m, 1], F[kb][m])
    gl = None
    kbed = field(F, "bedrock")
    if kbed is not None:
        g = m & (np.abs(F[kb] - F[kbed]) < 1.0)
        gl = P[g, 0].max() if g.any() else None
    return P[:, 0], P[:, 1], P[:, 2], u, zb, gl, what


def main():
    a = sys.argv[1:]

    def opt(name, conv, default):
        if name not in a:
            return default
        i = a.index(name)
        v = conv(a[i + 1])
        del a[i:i + 2]
        return v
    acc = opt("--a", float, A_DIM)
    every = opt("--every-km", float, 50.0)
    if len(a) != 1:
        raise SystemExit(__doc__)
    x, y, z, u, (zbx, zby, zbv), gl, what = (from_state(a[0]) if a[0].endswith(".npz") else from_vtu(a[0]))
    key = lambda xx, yy: np.round(xx).astype(np.int64) * 10 ** 8 + np.round(yy).astype(np.int64)
    kz = key(zbx, zby)
    zb_of = dict(zip(kz.tolist(), zbv.tolist()))
    k = key(x, y)
    zbn = np.array([zb_of.get(v, np.nan) for v in k.tolist()])
    ice = np.isfinite(zbn) & (z >= zbn - 0.5)
    k, x, z, u = k[ice], x[ice], z[ice], u[ice]
    o = np.lexsort((z, k))
    k, x, z, u = k[o], x[o], z[o], u[o]
    st = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    en = np.r_[st[1:], len(k)]
    cx = x[st]
    q = np.array([np.trapezoid(u[s:e], z[s:e]) if hasattr(np, "trapezoid") else np.trapz(u[s:e], z[s:e])
                  for s, e in zip(st, en)])
    # across y: the mean at each x
    xs, xi = np.unique(np.round(cx / 1e3, 3), return_inverse=True)
    qx = np.bincount(xi, weights=q) / np.bincount(xi)
    ref = acc * xs * 1e3
    r = np.where(ref > 0, qx / np.where(ref > 0, ref, 1), np.nan)
    print(f"{what}\n  accumulation a = {acc:.6g} m/yr; {len(st)} columns, {len(xs)} x positions; grounding line "
          + (f"{gl / 1e3:.2f} km" if gl is not None else "unknown"))
    print(f"  {'x (km)':>8s} {'q (m^2/yr)':>12s} {'a x (m^2/yr)':>13s} {'q / (a x)':>10s}")
    marks = sorted(set(np.searchsorted(xs, np.arange(every, xs.max(), every)).tolist() + [len(xs) - 1]))
    if gl is not None:
        marks = sorted(set(marks + [int(np.argmin(np.abs(xs - gl / 1e3)))]))
    for i in marks:
        if i < len(xs):
            tag = "  <- grounding line" if gl is not None and abs(xs[i] - gl / 1e3) < 0.6 else ""
            print(f"  {xs[i]:8.2f} {qx[i]:12.1f} {ref[i]:13.1f} {r[i]:10.4f}{tag}")
    good = xs > 20.0                             # away from the divide (q, a x -> 0)
    print(f"  q / (a x) beyond 20 km: min {np.nanmin(r[good]):.4f}, max {np.nanmax(r[good]):.4f}, "
          f"median {np.nanmedian(r[good]):.4f}")
    if gl is not None:
        i = int(np.argmin(np.abs(xs - gl / 1e3)))
        print(f"  at the grounding line: q_gl = {qx[i]:.1f} m^2/yr, a x_gl = {ref[i]:.1f}: ratio {r[i]:.4f}")
        schoof(gl, qx[i], acc)


def schoof(gl, q_gl, acc):
    """Schoof (2007) grounding-line flux for these parameters: Newtonian ice (n = 1, A = 1/(2 eta)), linear sliding
    (m = 1) with C = fcoeff (a temperate bed at the grounding line): q = [A (rho_i g)^2 (1 - rho_i/rho_w) / (4 C)]^(1/2)
    h^(5/2), h the flotation thickness on the bed table (case/marine_ics/bed_dim.xy, linear beyond its end); and the
    roots of q(h_f(x)) = a x on that bed."""
    from ess.case import eval_lua_parameters
    root = os.path.dirname(HERE)
    P = eval_lua_parameters(os.path.join(root, "case", "parameters_fine2x.lua"))
    A, C, rg, r = 1 / (2 * P["viscosity"]), P["fcoeff"], P["rhoi"] * P["gravity"], P["rhoi"] / P["rhow"]
    K = (A * rg ** 2 * (1 - r) / (4 * C)) ** 0.5
    b = np.loadtxt(os.path.join(root, "case", "marine_ics", "bed_dim.xy"))
    s_ = (b[-1, 1] - b[0, 1]) / b[-1, 0]
    bed = lambda x: np.where(x > b[-1, 0], b[0, 1] + s_ * x, np.interp(x, b[:, 0], b[:, 1]))
    hf = -bed(gl) / r
    qs = K * hf ** 2.5
    print(f"  Schoof: q = {K:.4g} h^2.5; flotation thickness at the grounding line {hf:.1f} m (bed table): Schoof's q "
          f"{qs:.0f} m^2/yr; the model's / Schoof's = {q_gl / qs:.3f}")
    xx = np.linspace(50e3, 1200e3, 230001)
    F = K * (-bed(xx) / r) ** 2.5 - acc * xx
    for j in np.where(np.diff(np.sign(F)))[0]:
        print(f"  Schoof's steady grounding line on this bed: {xx[j] / 1e3:.1f} km "
              f"({'stable' if F[j + 1] > F[j] else 'unstable'})")


if __name__ == "__main__":
    main()
