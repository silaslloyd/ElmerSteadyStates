#!/usr/bin/env python3
"""A surface bump on a land state, for the 3D free-surface time-step tests: Zs raised by a Gaussian
amp exp(-((x - x0)^2 + (y - y0)^2) / r^2), so the start is genuinely 3D (the land_margin_3d start is uniform across y)
and its relaxation (spreading in x and y) is the fast, stiff free-surface motion that limits explicit steps.

    python3 periodic_test/perturb_surface.py IN.npz OUT.npz [--amp 100] [--x0 0.5] [--y0 0.5] [--radius 4]

--amp m; --x0, --y0 as fractions of the ice's x and y extent; --radius km. Everything else in the state is kept.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import find_name, load_state, save_state   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--amp", type=float, default=100.0)
    ap.add_argument("--x0", type=float, default=0.5)
    ap.add_argument("--y0", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=4.0)
    a = ap.parse_args()

    s = load_state(a.src)
    allf = {**s.diag, **s.frozen}
    gx, X = allf[find_name(allf, "coordinate 1")]
    gy, Y = allf[find_name(allf, "coordinate 2")]
    Xd, Yd = dict(zip(gx.tolist(), X.tolist())), dict(zip(gy.tolist(), Y.tolist()))
    b = next(b for b in s.layout.blocks if b.name.lower() == "zs")
    sl = s.layout.slices()[b.name]
    g = b.gids.tolist()
    x = np.array([Xd[i] for i in g])
    y = np.array([Yd[i] for i in g])
    x0 = x.min() + a.x0 * np.ptp(x)
    y0 = y.min() + a.y0 * np.ptp(y)
    dz = a.amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (a.radius * 1e3) ** 2)
    v = s.x.copy()
    v[sl] = v[sl] + dz
    meta = dict(s.meta)
    meta["perturb_surface"] = f"amp {a.amp} m at x {x0 / 1e3:.1f} y {y0 / 1e3:.1f} km, radius {a.radius} km, from {a.src}"
    save_state(a.out, s.layout, v, s.p, frozen=s.frozen, diag=s.diag, **meta)
    print(f"Zs + {a.amp:g} m bump at x {x0 / 1e3:.1f}, y {y0 / 1e3:.1f} km (radius {a.radius:g} km; ice x "
          f"{x.min() / 1e3:.1f} .. {x.max() / 1e3:.1f}, y {y.min() / 1e3:.1f} .. {y.max() / 1e3:.1f} km; "
          f"{int(np.sum(dz > 1))} surface nodes raised > 1 m) -> {a.out}")


if __name__ == "__main__":
    main()
