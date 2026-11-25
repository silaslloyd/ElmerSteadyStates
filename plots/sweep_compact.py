#!/usr/bin/env python3
"""
The converged sweep states as ONE short text block to paste into a chat (a phone, no file transfer): for each sweep
point its patch length d, grounding line, and the surface Zs, the base Zb and the bed on a 5 km grid (whole metres),
as zlib + base64 in one line between the markers BEGIN / END.

    python3 plots/sweep_compact.py ROOT [--dx 5]       ROOT: the sweep folder (seg_*/sweep.csv + points/)
"""
import base64
import json
import sys
import zlib

import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
sys.path.insert(0, HERE)
from sia_start import column_profile, bed_profile  # noqa: E402
from ess.spectrum import gl_position  # noqa: E402
from ess.statefile import load_state  # noqa: E402
from sweep_rates import points  # noqa: E402


def main():
    a = sys.argv[1:]
    dx = 5.0
    if "--dx" in a:
        i = a.index("--dx")
        dx = float(a[i + 1])
        del a[i:i + 2]
    if not a:
        raise SystemExit(__doc__)
    pts = {}
    for r in a:
        pts.update(points(r))
    out = []
    xg = None
    for d in sorted(pts, reverse=True):
        p, its = pts[d]
        s = load_state(p)
        xs, zs = column_profile(s, "zs")
        try:
            xb, zb = column_profile(s, "zb")
        except StopIteration:
            xb, zb = xs, np.full_like(xs, np.nan)
        xr, zr = bed_profile(s)
        if xg is None:
            xg = np.arange(0.0, xs.max() / 1e3 + 1e-9, dx)
        f = lambda x, z: [int(round(v)) if np.isfinite(v) else None for v in np.interp(xg * 1e3, x, z)]
        out.append({"d": d, "its": its, "gl": round(gl_position(s.frozen, s.diag) / 1e3, 2),
                    "zs": f(xs, zs), "zb": f(xb, zb), "bed": f(xr, zr)})
    blob = base64.b64encode(zlib.compress(json.dumps({"dx": dx, "pts": out}, separators=(",", ":")).encode(), 9)).decode()
    print(f"{len(out)} states, {len(xg)} x nodes, {len(blob)} characters")
    print("BEGIN")
    print(blob)
    print("END")


if __name__ == "__main__":
    main()
