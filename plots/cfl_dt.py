#!/usr/bin/env python3
"""The largest time step that keeps the surface CFL at or below 1 on a mesh, from a state on it (e.g. the start state
of a remeshed mesh, which carries the velocity interpolated onto it): CFL = |u_h| dt / h at each surface node, h the
shortest edge at the node (as elmer/StepCheck.F90), so dt_max = min h / |u_h|.

    python3 plots/cfl_dt.py MESHDIR STATE.npz [MESHDIR2 STATE2.npz ...]

Also: where that minimum is, and the dt at which only 0.1 % / 1 % of the nodes would exceed CFL 1.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.remesh import read_footprint  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402


def one(meshdir, state):
    _, x, y, tri = read_footprint(meshdir)
    nb = len(x)
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
    e = np.unique(np.sort(e, 1), axis=0)
    L = np.hypot(x[e[:, 0]] - x[e[:, 1]], y[e[:, 0]] - y[e[:, 1]])
    h = np.full(nb, np.inf)
    np.minimum.at(h, e[:, 0], L)
    np.minimum.at(h, e[:, 1], L)
    s = load_state(state)
    f = {**s.diag, **s.frozen}
    g1, u = f[find_name(f, "velocity 1")]
    g2, v = f[find_name(f, "velocity 2")]
    if not np.array_equal(g1, g2):
        raise SystemExit("velocity 1 and 2 on different nodes")
    lev = (g1 - 1) // nb
    top = lev == lev.max()                              # the surface: the top level of the ice
    col = (g1[top] - 1) % nb
    sp = np.full(nb, np.nan)
    sp[col] = np.hypot(u[top], v[top])
    ok = np.isfinite(sp) & (sp > 1e-6)
    dt = h[ok] / sp[ok]
    k = np.argmin(dt)
    idx = np.flatnonzero(ok)[k]
    q1, q01 = np.quantile(dt, 0.01), np.quantile(dt, 0.001)
    print(f"{os.path.basename(os.path.normpath(meshdir))}: {nb} nodes; CFL 1 needs dt <= {dt[k]:.2f} yr "
          f"(at x {x[idx] / 1e3:.1f} y {y[idx] / 1e3:.1f} km: speed {sp[idx]:.0f} m/yr, edge {h[idx]:.0f} m); "
          f"0.1% of nodes above CFL 1 at dt {q01:.2f} yr, 1% at {q1:.2f} yr; CFL at dt 10: {10 / dt[k]:.1f}, "
          f"at dt 2: {2 / dt[k]:.1f}")


def main():
    a = sys.argv[1:]
    if len(a) < 2 or len(a) % 2:
        raise SystemExit(__doc__)
    for m, st in zip(a[0::2], a[1::2]):
        one(m, st)


if __name__ == "__main__":
    main()
