#!/usr/bin/env python3
"""The bed along the radius from a frames file (plots/vtu_frames.py: base_bed) or a state file (frozen "bedrock"):
min / mean / max over the angle in radius bins, and the jump from one bin to the next (a step shows as a jump far
above the slope x bin width).

    python3 plots/bed_profile.py FILE [--r KM KM] [--bin KM]
"""
import os
import sys

import numpy as np


def main():
    a = sys.argv[1:]
    lo, hi, bw = 400.0, 1600.0, 2.0
    if "--r" in a:
        k = a.index("--r"); lo, hi = float(a[k + 1]), float(a[k + 2]); del a[k:k + 3]
    if "--bin" in a:
        k = a.index("--bin"); bw = float(a[k + 1]); del a[k:k + 2]
    z = np.load(a[0], allow_pickle=False)
    if "base_bed" in z.files:
        x, y, b = z["base_x"], z["base_y"], z["base_bed"]
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from ess.statefile import find_name, load_state
        s = load_state(a[0])
        f = {**s.diag, **s.frozen}
        g, b = s.frozen[find_name(s.frozen, "bedrock")]
        look = lambda n: dict(zip(*f[find_name(f, n)]))
        cx, cy = look("coordinate 1"), look("coordinate 2")
        x = np.array([cx[i] for i in g]); y = np.array([cy[i] for i in g])
    r = np.hypot(x, y) / 1e3
    edges = np.arange(lo, hi + bw, bw)
    prev = None
    print(f"{a[0]}: bed by radius ({bw:g} km bins)\n{'r (km)':>8} {'n':>6} {'min':>9} {'mean':>9} {'max':>9} {'jump':>8}")
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (r >= e0) & (r < e1)
        if not m.any():
            continue
        mu = b[m].mean()
        print(f"{0.5 * (e0 + e1):8.1f} {m.sum():6d} {b[m].min():9.2f} {mu:9.2f} {b[m].max():9.2f} "
              f"{'' if prev is None else f'{mu - prev:8.2f}'}")
        prev = mu


if __name__ == "__main__":
    main()
