#!/usr/bin/env python3
"""Compare the 3D strip runs of fssa3d/sweep.sh: the response to the friction dip for each stabilisation / scheme.

    python3 fssa3d/ess_compare.py [--work work/flowline_gl_3d] [--start START.npz] [--x0 600]

For every fssa_<TAG>.npz: the grounding line along the centre (the dip) and at the wall (y = 0), the change of Zs and Zb
from the start state (rms over x >= x0 km, centre line and wall), the largest cross-flow difference in Zs (centre - wall),
and the grid-scale roughness of Zs along x and y (second differences: what an unstable free surface grows). Each dip run
is also differenced against its control (same stabilisation, no dip) when there is one.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import load_state, find_name   # noqa: E402
from ess.masks import _on                           # noqa: E402


def fields(path):
    s = load_state(path)
    blk = s.layout.split(s.x)
    allf = dict(s.diag)
    allf.update(s.frozen)
    gx, X = allf[find_name(allf, "coordinate 1")]
    gy, Y = allf[find_name(allf, "coordinate 2")]
    out = {"t": float(s.meta.get("time", np.nan))}
    for name in ("Zs", "Zb"):
        b = next((b for b in s.layout.blocks if b.name.lower() == name.lower()), None)
        if b is None:
            continue
        out[name] = (_on(gx, X, b.gids), _on(gy, Y, b.gids), blk[b.name])
    try:
        g, gm = s.frozen[find_name(s.frozen, "GroundedMask")]
        out["mask"] = (_on(gx, X, g), _on(gy, Y, g), gm)
    except KeyError:
        pass
    return out


def line(F, name, yfrac, x0):
    """the values along x on the node line nearest y = yfrac * (y range), x >= x0"""
    x, y, v = F[name]
    y0 = y.min() + yfrac * np.ptp(y)
    yl = y[np.argmin(np.abs(y - y0))]
    m = (np.abs(y - yl) < 1e-6 + 1e-9 * abs(yl)) & (x >= x0)
    o = np.argsort(x[m])
    return x[m][o], v[m][o]


def gl(F, yfrac):
    x, y, m = F["mask"]
    y0 = y.min() + yfrac * np.ptp(y)
    yl = y[np.argmin(np.abs(y - y0))]
    sel = (np.abs(y - yl) < 1e-6 + 1e-9 * abs(yl)) & (m > -0.5)
    return float(x[sel].max() / 1e3) if sel.any() else float("nan")


def rough(F, x0):
    x, y, z = F["Zs"]
    m = x >= x0
    xs, ys = np.unique(np.round(x[m])), np.unique(np.round(y[m], 1))
    g = np.full((len(ys), len(xs)), np.nan)
    g[np.searchsorted(ys, np.round(y[m], 1)), np.searchsorted(xs, np.round(x[m]))] = z[m]
    dx = np.abs(np.diff(g, 2, axis=1))
    dy = np.abs(np.diff(g, 2, axis=0))
    return (float(np.nanmax(dx)) if dx.size else np.nan), (float(np.nanmax(dy)) if dy.size else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work/flowline_gl_3d")
    ap.add_argument("--start", default=None)
    ap.add_argument("--x0", type=float, default=600.0, help="km")
    a = ap.parse_args()
    x0 = a.x0 * 1e3
    S = fields(a.start or os.path.join(a.work, "start_flowline.npz"))
    print(f"start: GL centre {gl(S, .5):.2f} km, wall {gl(S, 0):.2f} km")
    runs = {os.path.basename(p)[5:-4]: fields(p) for p in sorted(glob.glob(os.path.join(a.work, "fssa_*.npz")))
            if "frames" not in p and not p.endswith("fssa_start.npz")}
    if not runs:
        sys.exit("no fssa_*.npz in " + a.work)

    def drift(F, G, name, yfrac):
        x, v = line(F, name, yfrac, x0)
        xs, vs = line(G, name, yfrac, x0)
        d = v - np.interp(x, xs, vs)
        return float(np.sqrt(np.mean(d ** 2))), float(d[np.argmax(np.abs(d))])

    print(f"{'run':40s} {'t(yr)':>6s} {'GL ctr':>8s} {'GL wall':>8s} {'dZs ctr rms/max':>16s} {'dZs wall':>16s} "
          f"{'dZb ctr rms':>11s} {'cross-flow Zs':>13s} {'rough x/y (m)':>14s}")
    for tag, F in runs.items():
        rc, mc = drift(F, S, "Zs", .5)
        rw, mw = drift(F, S, "Zs", 0.0)
        zb = drift(F, S, "Zb", .5)[0] if "Zb" in F else float("nan")
        xc, vc = line(F, "Zs", .5, x0)
        xw, vw = line(F, "Zs", 0.0, x0)
        cross = float(np.max(np.abs(vc - np.interp(xc, xw, vw))))
        rx, ry = rough(F, x0)
        print(f"{tag:40s} {F['t']:6.0f} {gl(F, .5):8.2f} {gl(F, 0):8.2f} {rc:7.2f}/{mc:+7.1f} {rw:7.2f}/{mw:+7.1f} "
              f"{zb:11.2f} {cross:13.2f} {rx:6.2f}/{ry:6.2f}")
    ctl = {t: F for t, F in runs.items() if t.startswith("control_")}
    if ctl:
        print("\nagainst the control of the same scheme (dip - control):")
        for tag, F in runs.items():
            if tag.startswith("control_"):
                continue
            key = "control_" + tag.split("_", 2)[2]
            key = next((k for k in ctl if k.endswith(tag.split("_", 2)[2])), None)
            if key is None:
                continue
            rc, mc = drift(F, ctl[key], "Zs", .5)
            rw, mw = drift(F, ctl[key], "Zs", 0.0)
            print(f"{tag:40s} vs {key:34s} dGL centre {gl(F, .5) - gl(ctl[key], .5):+.2f} km, wall "
                  f"{gl(F, 0) - gl(ctl[key], 0):+.2f} km; Zs centre rms {rc:.2f} m (max {mc:+.1f}), wall rms {rw:.2f} m")


if __name__ == "__main__":
    main()
