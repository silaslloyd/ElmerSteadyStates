#!/usr/bin/env python3
"""Where the modes of an `ess stability` result live: for each mode, mu, the growth rate sigma and its e-folding time,
the share of the mode (in the scaled variables Newton uses) in the temperature of the ice, the temperature of the
bedrock, Zs and Zb, and the x of each part's largest component.

    python3 plots/stability_modes.py work/<case>/stability/<name>/stability.npz [--modes 6]

Unstable: |mu| > 1 (sigma > 0). A stable mode decays by e in 1/|sigma|.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import StateLayout  # noqa: E402


def main():
    args = sys.argv[1:]
    nm = 6
    if "--modes" in args:
        k = args.index("--modes")
        nm = int(args[k + 1])
        del args[k:k + 2]
    if len(args) != 1:
        raise SystemExit(__doc__)
    z = np.load(args[0])
    lay = StateLayout.from_npz(z)
    mu, sigma, V = z["mu"], z["sigma"], z["vectors"]
    conv = z["converged"]
    T = float(z["dt"]) * int(z["n_steps"])
    rec = lambda pre: {k[len(pre):]: (z[k], z[k.replace("gids", "vals")]) for k in z.files if k.startswith(pre)}
    diag, frozen = rec("diag_gids::"), rec("frozen_gids::")
    look = lambda d, name: next(v for k, v in d.items() if k.lower() == name)
    gx, X = look(diag, "coordinate 1")
    gz, Z = look(diag, "coordinate 3")
    gb, B = look(frozen, "bedrock")
    xo, zo, bo = (dict(zip(g.tolist(), v)) for g, v in ((gx, X), (gz, Z), (gb, B)))
    sl = lay.slices()
    print(f"{args[0]}: T = {T:g} yr, drift rms(Phi(x*) - x*) = {float(z['drift']):.2e}")
    for i in range(min(nm, len(mu))):
        v = V[:, i]
        parts = {}
        for b in lay.blocks:
            vb = np.abs(v[sl[b.name]]) / b.scale                     # scaled, as Newton sees it
            xs = np.array([xo.get(g, np.nan) for g in b.gids]) / 1e3
            if b.name.lower() == "temperature":
                rock = np.array([zo.get(g, np.nan) < bo.get(g, np.nan) - 0.5 for g in b.gids])
                for nm_, m in (("T ice", ~rock), ("T rock", rock)):
                    parts[nm_] = (vb[m], xs[m])
            else:
                parts[b.name] = (vb, xs)
        tot = sum(float(np.sum(p[0] ** 2)) for p in parts.values()) or 1.0
        s = sigma[i]
        ef = (f"grows x e in {1 / s.real:,.0f} yr" if s.real > 0 else f"decays x 1/e in {-1 / s.real:,.0f} yr")
        osc = f", period {2 * np.pi / abs(s.imag):,.0f} yr" if abs(s.imag) > 1e-12 else ""
        print(f"mode {i}: |mu| {abs(mu[i]):.6f}  sigma {s.real:+.3e}/yr ({ef}{osc})"
              f"{'' if conv[i] else '  NOT converged'}")
        for nm_, (vb, xs) in parts.items():
            if vb.size:
                j = int(np.nanargmax(vb))
                print(f"    {nm_:7s} {np.sum(vb ** 2) / tot:6.1%}  largest at x = {xs[j]:.1f} km")


if __name__ == "__main__":
    main()
