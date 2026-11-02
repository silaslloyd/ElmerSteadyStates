#!/usr/bin/env python3
"""Compare the end states of flowline free-surface runs (jobs/flowline_fssa.cmd: work/marine_flowline/fssa_<TAG>.npz)
with a reference (the smallest step): grounding line, surface and base differences, and the surface roughness (the
node-to-node zig-zag that an unstable free surface grows: the largest |second difference| of Zs along x, and where).

    python3 periodic_test/fssa_compare.py REF.npz RUN.npz [RUN2.npz ...] [--x0 600]
--x0 KM: roughness and differences only beyond this x (default 600 km: the grounding-line region and the shelf)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import load_state, find_name
from ess.masks import _on

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("runs", nargs="+"); ap.add_argument("--x0", type=float, default=600.0)
a = ap.parse_args()


def profile(path):
    s = load_state(path)
    blk = s.layout.split(s.x)
    allf = dict(s.diag); allf.update(s.frozen)
    gx, X = allf[find_name(allf, "coordinate 1")]
    out = {}
    for name in ("Zs", "Zb"):
        b = next(b for b in s.layout.blocks if b.name.lower() == name.lower())
        x = _on(gx, X, b.gids)
        xr = np.round(x, 0)
        ux = np.unique(xr)
        v = np.array([blk[b.name][xr == u].mean() for u in ux])     # the strip's width averaged (one element wide)
        out[name] = (ux, v)
    g, gm = s.frozen[find_name(s.frozen, "GroundedMask")]
    xg = _on(gx, X, g)
    out["gl"] = float(xg[gm > -0.5].max())
    out["t"] = float(s.meta.get("time", np.nan))
    return out


def rough(x, z, x0):
    m = x >= x0 * 1e3
    x, z = x[m], z[m]
    if len(x) < 3:
        return np.nan, np.nan
    d2 = np.abs(z[2:] - 2 * z[1:-1] + z[:-2])
    k = int(np.argmax(d2))
    return float(d2[k]), float(x[k + 1] / 1e3)


R = profile(a.ref)
rr, rx = rough(*R["Zs"], a.x0)
print(f"reference {os.path.basename(a.ref)}: t {R['t']:g} yr, grounding line {R['gl'] / 1e3:.2f} km, "
      f"surface roughness max |d2 Zs| {rr:.3g} m at {rx:.1f} km")
for p in a.runs:
    try:
        P = profile(p)
    except Exception as e:
        print(f"{os.path.basename(p)}: not readable ({e})")
        continue
    line = f"{os.path.basename(p)}: t {P['t']:g} yr, grounding line {P['gl'] / 1e3:.2f} km ({(P['gl'] - R['gl']) / 1e3:+.2f})"
    for name in ("Zs", "Zb"):
        x, v = P[name]
        xr_, vr = R[name]
        m = x >= a.x0 * 1e3
        d = v[m] - np.interp(x[m], xr_, vr)
        k = int(np.argmax(np.abs(d))) if len(d) else 0
        line += f"; {name} rms {np.sqrt(np.mean(d ** 2)):.2f} m, max {d[k]:+.1f} m at {x[m][k] / 1e3:.1f} km" if len(d) else ""
    r, xr2 = rough(*P["Zs"], a.x0)
    line += f"; roughness {r:.3g} m at {xr2:.1f} km ({r / rr:.1f}x the reference)" if rr > 0 else f"; roughness {r:.3g} m"
    print(line)
