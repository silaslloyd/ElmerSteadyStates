#!/usr/bin/env python3
"""The drop test's start (configs/spread_drop_*.toml): a state (normally the config's state0.npz from `ess init`, the
uniform layer spread_thickness on the flat bed) with a drop added on top about the box centre (or source_x_km,
source_y_km):
    drop_shape = "smooth" (default): H = spread_thickness + drop_height_m (1 - (r / R)^2)^2  for r < R = drop_radius_km
                                     (smooth at its edge; volume pi R^2 drop_height / 3);
    drop_shape = "huppert":          H = spread_thickness + drop_height_m (1 - (r / R)^2)^(1/3)
                                     (Huppert's (1982) similarity shape: the drop on a dry no-slip bed is this shape at
                                     every time; volume 3/4 pi R^2 drop_height).
The temperature is left as it is (the isothermal test), or set to [case] initial_temperature everywhere (the cooling
drop: the drop and the thin layer start hot, the surface cools them).

    python3 jobs/spread_drop.py CONFIG STATE.npz OUT.npz
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.statefile import find_name, load_state, save_state  # noqa: E402


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    cfgf, src, out = sys.argv[1:]
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    c = tomllib.load(open(cfgf, "rb"))["case"]
    L, W = float(c["spread_length_km"]) * 1e3, float(c["width"])
    xc, yc = float(c.get("source_x_km", L / 2e3)) * 1e3, float(c.get("source_y_km", W / 2e3)) * 1e3
    A, R = float(c["drop_height_m"]), float(c["drop_radius_km"]) * 1e3
    s = load_state(src)
    allc = {**s.diag, **s.frozen}
    X, Y = (dict(zip(*allc[find_name(allc, f"coordinate {k}")])) for k in (1, 2))
    blk = {b.name.lower(): b for b in s.layout.blocks}
    bz = blk["zs"]
    g = bz.gids
    r = np.hypot(np.array([X[k] for k in g]) - xc, np.array([Y[k] for k in g]) - yc)
    shape = str(c.get("drop_shape", "smooth"))
    if shape == "huppert":
        add = A * np.cbrt(np.clip(1 - (r / R) ** 2, 0, None))
        vol = 0.75 * np.pi * R ** 2 * A
    else:
        add = A * np.clip(1 - (r / R) ** 2, 0, None) ** 2
        vol = np.pi * R ** 2 * A / 3
    parts = s.layout.split(s.x)
    parts[bz.name] = parts[bz.name] + add
    if c.get("initial_temperature") is not None:     # the cooling drop: everything starts hot (the layer too)
        bT = next(b for b in s.layout.blocks if b.name.lower() == "temperature")
        parts[bT.name] = np.full_like(parts[bT.name], float(c["initial_temperature"]))
    x = np.concatenate([parts[b.name] for b in s.layout.blocks])
    save_state(out, s.layout, x, s.p, s.frozen, s.diag, **{**s.meta, "time": 0.0, "drop_height_m": A,
                                                            "drop_radius_km": R / 1e3})
    print(f"drop ({shape}): {A:g} m high, radius {R / 1e3:g} km at ({xc / 1e3:g}, {yc / 1e3:g}) km; volume "
          f"{vol:.3g} m^3; {int((add > 0).sum())} surface nodes raised -> {out}")


if __name__ == "__main__":
    main()
