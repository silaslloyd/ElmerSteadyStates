#!/usr/bin/env python3
"""The spreading test: how far the current has got, per state, in numbers (no page). For each state (model-time
order): the time, the volume above the background layer (the integral of h = H - H0 over the footprint, on the
surface nodes' Delaunay triangles), the area and the x / y extent where h > --dh (default 0.02 m), the thickest h and
the fastest surface horizontal speed.

    python3 plots/spread_extent.py STATE.npz [STATE2.npz ...] [--config configs/spread_basalt.toml] [--dh 0.02]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.statefile import find_name, load_state  # noqa: E402


def arg(args, name, default, conv=str):
    if name in args:
        k = args.index(name)
        v = conv(args[k + 1])
        del args[k:k + 2]
        return v
    return default


def fmt_t(t):
    if not np.isfinite(t) or t == 0:
        return "0"
    if t >= 1:
        return f"{t:.3g} yr"
    if t >= 1 / 365.25:
        return f"{t * 365.25:.3g} d"
    if t >= 1 / 8766:
        return f"{t * 8766:.3g} h"
    if t >= 1 / 525960:
        return f"{t * 525960:.3g} min"
    return f"{t * 31557600:.3g} s"


def main():
    args = sys.argv[1:]
    cfgf = arg(args, "--config", os.path.join(ROOT, "configs", "spread_basalt.toml"))
    dh = arg(args, "--dh", 0.02, float)
    if not args:
        raise SystemExit(__doc__)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    c = tomllib.load(open(cfgf, "rb"))["case"]
    H0 = float(c.get("spread_thickness", 100.0))
    from scipy.spatial import Delaunay
    rows = []
    for p in args:
        s = load_state(p)
        allc = {**s.diag, **s.frozen}
        X = dict(zip(*allc[find_name(allc, "coordinate 1")]))
        Y = dict(zip(*allc[find_name(allc, "coordinate 2")]))
        B = dict(zip(*allc[find_name(allc, "bedrock")]))
        blk = {b.name.lower(): b for b in s.layout.blocks}
        gs = blk["zs"].gids
        zs = s.layout.split(s.x)[blk["zs"].name]
        x = np.array([X[g] for g in gs])
        y = np.array([Y[g] for g in gs])
        h = zs - np.array([B[g] for g in gs]) - H0
        try:
            V = [dict(zip(*allc[find_name(allc, f"velocity {k}")])) for k in (1, 2)]
            U = np.hypot(*[np.array([v_.get(g, np.nan) for g in gs], float) for v_ in V])
        except KeyError:
            U = np.full(len(gs), np.nan)
        tri = Delaunay(np.c_[x, y]).simplices
        a = 0.5 * np.abs((x[tri[:, 1]] - x[tri[:, 0]]) * (y[tri[:, 2]] - y[tri[:, 0]])
                         - (x[tri[:, 2]] - x[tri[:, 0]]) * (y[tri[:, 1]] - y[tri[:, 0]]))
        hp = np.clip(h, 0, None)
        vol = float(np.sum(a * hp[tri].mean(axis=1)))
        wet = h > dh
        area = float(np.sum(a[(h[tri] > dh).all(axis=1)]))
        ext = (f"x {x[wet].min():6.2f} .. {x[wet].max():6.2f} m, y {y[wet].min():6.2f} .. {y[wet].max():6.2f} m"
               if wet.any() else "nowhere")
        rows.append((float(s.meta.get("time", np.nan)), os.path.basename(p), vol, area, ext, float(h.max()),
                     float(np.nanmax(U)) if np.isfinite(U).any() else np.nan))
    rows.sort()
    for t, name, vol, area, ext, hmax, umax in rows:
        print(f"{name}: t {fmt_t(t):>9}  volume {vol:7.3f} m^3  area(h > {dh:g} m) {area:7.2f} m^2  {ext}  "
              f"h max {hmax:.3f} m  speed max {umax / 31557600:.3g} m/s")


if __name__ == "__main__":
    main()
