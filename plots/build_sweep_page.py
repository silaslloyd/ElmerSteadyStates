#!/usr/bin/env python3
"""
The Bed Sweep Profiles page (the same page, plots/sweep_profiles_tpl.html) built from saved states, as one HTML file.

    python3 plots/build_sweep_page.py OUT.html ROOT [ROOT2 ...]            ROOT: a sweep folder (seg_*/sweep.csv + points/)
    python3 plots/build_sweep_page.py OUT.html LABEL=DIR [LABEL2=DIR2 ...]  DIR: a branch run's points folder (state_NNNN.npz)

For a branch run the states are in the order given (labels in the order given, each by state number); the slider's
first state is the first one, the "d" column counts down from the number of states. The page is otherwise unchanged.
"""
import glob
import json
import os
import re
import sys

import numpy as np

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
    if len(a) < 2:
        raise SystemExit(__doc__)
    out, srcs, dx = a[0], a[1:], 5.0
    items = []                         # (d, its, file)
    if any("=" in s for s in srcs):
        files = []
        for s in srcs:
            _, _, folder = s.partition("=")
            fs = glob.glob(os.path.join(folder, "state_*.npz"))
            fs.sort(key=lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[-1]))
            files += fs
        n = len(files)
        items = [(float(n - i), 0, f) for i, f in enumerate(files)]
    else:
        pts = {}
        for r in srcs:
            pts.update(points(r))
        items = [(d, pts[d][1], pts[d][0]) for d in sorted(pts, reverse=True)]
    if not items:
        raise SystemExit("no states found")
    res, xg = [], None
    for d, its, p in items:
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
        res.append({"d": d, "its": its, "gl": round(gl_position(s.frozen, s.diag) / 1e3, 2),
                    "zs": f(xs, zs), "zb": f(xb, zb), "bed": f(xr, zr)})
    tpl = open(os.path.join(HERE, "sweep_profiles_tpl.html")).read()
    html = tpl.replace("/*DATA*/null", json.dumps({"dx": dx, "pts": res}, separators=(",", ":")))
    open(out, "w").write(html)
    print(f"{len(res)} states -> {out}")


if __name__ == "__main__":
    main()
