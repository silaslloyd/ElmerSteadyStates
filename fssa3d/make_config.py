#!/usr/bin/env python3
"""A 3D marine config from a converged flowline config: the same physics, bed and mesh along x, but WIDTH across
with free-slip walls and Y_ELEMENTS elements across, and an optional friction dip in the middle of the width
([case] slide_ydip, slide_ydip_width; ess/elmer.py). The flowline state is extruded laterally onto it (`ess transfer`),
so the 3D run starts from a y-uniform steady state; the friction dip is the perturbation.

    python3 fssa3d/make_config.py work/flowline_gl_fine/refined/pipe1.toml configs/flowline_gl_3d.toml \\
        [--width 20000] [--y-elements 20] [--min-res 500] [--dip 0.0] [--dip-width 2] [--work work/flowline_gl_3d]

--front-km X: cut the shelf, the front at x = X km (the 3D tests use 50 km of shelf: fssa3d/prep.cmd SHELF_KM).
--min-res M: no mesh cell smaller than M m along x (the flowline's 250 m band is expensive in 3D: 1e6+ nodes).
--dip: written into the config (default 0: uniform friction; the runs set ESS_SLIDE_YDIP to switch the dip on).
"""
import argparse
import ast
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--width", type=float, default=20000.0)
    ap.add_argument("--y-elements", type=int, default=20)
    ap.add_argument("--min-res", type=float, default=500.0)
    ap.add_argument("--dip", type=float, default=0.0)
    ap.add_argument("--dip-width", type=float, default=2.0, help="km")
    ap.add_argument("--work", default="work/flowline_gl_3d")
    ap.add_argument("--front-km", type=float, default=None,
                    help="cut the shelf: the calving front at this x (km), res_points truncated there")
    a = ap.parse_args()
    t = open(a.src).read()

    def sub(key, val, t):
        n = len(re.findall(rf"^{key}\s*=", t, re.M))
        if n != 1:
            raise SystemExit(f"{a.src}: expected one '{key} =' line, found {n}")
        return re.sub(rf"^({key}\s*=\s*)[^#\n]*", lambda m: m.group(1) + val + " ", t, flags=re.M)

    t = sub("work_dir", f'"{a.work}"', t)
    t = sub("width", repr(a.width), t)
    t = sub("y_elements", str(a.y_elements), t)
    t = sub("periodic_sides", "false", t)
    m = re.search(r"^res_points\s*=\s*(\[.*\])", t, re.M)
    if m:
        pts = [[x, max(r, a.min_res)] for x, r in ast.literal_eval(m.group(1))]
        if a.front_km is not None:
            import numpy as np
            xs, rs = [p[0] for p in pts], [p[1] for p in pts]
            rf = float(np.interp(a.front_km, xs, rs))
            pts = [p for p in pts if p[0] < a.front_km - 1e-6] + [[a.front_km, rf]]
        t = t.replace(m.group(0), "res_points = " + repr(pts).replace("],", "],"))
    if a.front_km is not None:
        t = sub("front_km", repr(float(a.front_km)), t)
    t = sub("slide_ybump", "0.0", t)
    t = re.sub(r"^(slide_ybump\s*=.*)$", rf"\1\nslide_ydip = {a.dip!r}                  # friction dip in the middle of the width (fraction)"
               rf"\nslide_ydip_width = {a.dip_width!r}              # km, e-folding half width", t, count=1, flags=re.M)
    t = "# 3D strip made by fssa3d/make_config.py from " + a.src + "\n" + t
    open(a.out, "w").write(t)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
