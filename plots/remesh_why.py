#!/usr/bin/env python3
"""Why the remesh trigger fires (elmer/RemeshTrigger.F90), from a state on the current mesh (e.g. jobs/latest_state.py's
checkpoint state, which carries the mesh's Remesh Distance): which grounding-line nodes (|GroundedMask| < 0.5) are
closer to coarse mesh than the GL safety distance, which cold/temperate-transition nodes closer than the transition
safety distance, and WHERE they are (by x bands, against where the rest of each curve is).

    python3 plots/remesh_why.py STATE.npz MESHDIR [--gl-safety 5000] [--ct-safety 1000] [--band-km 10]

MESHDIR: the mesh the state is on (work/<dir>/case/<mesh>, the MESH= line of jobs/latest_state.py).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.remesh import base_values, read_footprint  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402


def bands(x, sel, w):
    """counts of the selected nodes per x band of w km: 'x0..x1 km: n'"""
    if not sel.any():
        return "none"
    b = np.floor(x[sel] / 1e3 / w).astype(int)
    u, c = np.unique(b, return_counts=True)
    return ", ".join(f"{k * w:g}..{(k + 1) * w:g} km: {n}" for k, n in zip(u, c))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state")
    ap.add_argument("meshdir")
    ap.add_argument("--gl-safety", type=float, default=5000.0)
    ap.add_argument("--ct-safety", type=float, default=1000.0)
    ap.add_argument("--band-km", type=float, default=10.0)
    a = ap.parse_args()
    _, x, y, tri = read_footprint(a.meshdir)
    nb = len(x)
    s = load_state(a.state)
    f = {**s.diag, **s.frozen}
    gm = base_values(f[find_name(f, "groundedmask")], nb)
    ct = base_values(f[find_name(f, "coldtempmask")], nb)
    d = base_values(f[find_name(f, "remesh distance")], nb)
    if np.isnan(gm).all() or np.isnan(d).all():
        raise SystemExit("the state is not on this mesh (no GroundedMask / Remesh Distance on its base nodes)")
    print(f"state t = {float(s.meta.get('time', np.nan)):g} yr; mesh {os.path.basename(os.path.normpath(a.meshdir))}: "
          f"{nb} footprint nodes")
    gl = np.abs(np.nan_to_num(gm, nan=-9)) < 0.5
    near = gl & (d < a.gl_safety)
    print(f"\ngrounding line: {gl.sum()} nodes (x {x[gl].min() / 1e3:.1f} .. {x[gl].max() / 1e3:.1f} km, median "
          f"{np.median(x[gl]) / 1e3:.1f}); {near.sum()} within {a.gl_safety:g} m of coarse mesh")
    if near.any():
        print(f"  those by x: {bands(x, near, a.band_km)}")
        print(f"  their y: {y[near].min() / 1e3:.1f} .. {y[near].max() / 1e3:.1f} km; distance to coarse "
              f"{d[near].min():.0f} .. {d[near].max():.0f} m")
        far = np.abs(x[near] - np.median(x[gl])) > 20e3
        if far.any():
            print(f"  {far.sum()} of them more than 20 km from the main grounding line's median x: separate pieces "
                  f"(floating patches inland, or grounding points on the shelf)")
    # the transition: grounded base elements with both a cold and a temperate node (as the trigger)
    grounded = np.nan_to_num(gm, nan=-1) > -0.5
    temp = np.nan_to_num(ct, nan=-1) > 0
    ok = grounded & ~np.isnan(ct)
    isT = np.zeros(nb, bool)
    for t in tri:
        m = ok[t]
        if m.sum() and temp[t][m].any() and (~temp[t][m]).any():
            isT[t[m]] = True
    nearT = isT & (d < a.ct_safety)
    frac = nearT.sum() / max(isT.sum(), 1)
    print(f"\ncold/temperate transition: {isT.sum()} nodes; {nearT.sum()} within {a.ct_safety:g} m of coarse mesh "
          f"({100 * frac:.1f} %; the trigger fires above 5 %)")
    if nearT.any():
        print(f"  those by x: {bands(x, nearT, a.band_km)}")
        print(f"  all transition nodes by x: {bands(x, isT, a.band_km)}")
    print(f"\ntemperate grounded bed: x {x[grounded & temp].min() / 1e3:.1f} .. {x[grounded & temp].max() / 1e3:.1f} km"
          if (grounded & temp).any() else "\nno temperate grounded bed")


if __name__ == "__main__":
    main()
