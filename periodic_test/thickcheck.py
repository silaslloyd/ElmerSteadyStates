#!/usr/bin/env python3
"""Ice thickness of two runs on the same mesh, per column (Zs - Zb): the largest differences and the thinnest
floating ice, with where they are. python3 periodic_test/thickcheck.py REF.npz TEST.npz [--xmin km]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gl_ess import _state_fields, _field   # noqa: E402


def thick(path):
    f = _state_fields(path)
    g, X = _field(f, "coordinate 1")
    Xd = dict(zip(g.tolist(), X.tolist()))
    Yd = dict(zip(*_field(f, "coordinate 2")))
    col = lambda i: (round(Xd[i], 1), round(Yd[i], 1))
    zs = {col(i): v for i, v in zip(*[a.tolist() for a in _field(f, "Zs")]) if i in Xd}
    zb = {col(i): v for i, v in zip(*[a.tolist() for a in _field(f, "Zb")]) if i in Xd}
    gm = {col(i): v for i, v in zip(*[a.tolist() for a in _field(f, "GroundedMask")]) if i in Xd}
    return {k: (zs[k], zb[k], gm.get(k, 0.0)) for k in zs.keys() & zb.keys()}


ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("test")
ap.add_argument("--xmin", type=float, default=700.0, help="km: only columns downstream of this")
a = ap.parse_args()
r, t = thick(a.ref), thick(a.test)
keys = [k for k in r.keys() & t.keys() if k[0] >= a.xmin * 1e3]
dH = sorted(keys, key=lambda k: -abs((t[k][0] - t[k][1]) - (r[k][0] - r[k][1])))
print("  largest thickness differences (x km, y km: H ref, H test; Zs ref/test; Zb ref/test):")
for k in dH[:6]:
    print(f"    {k[0]/1e3:8.2f} {k[1]/1e3:6.2f}:  H {r[k][0]-r[k][1]:7.1f} {t[k][0]-t[k][1]:7.1f}   "
          f"Zs {r[k][0]:7.1f} {t[k][0]:7.1f}   Zb {r[k][1]:7.1f} {t[k][1]:7.1f}")
fl = [k for k in keys if t[k][2] < -0.5]
thin = sorted(fl, key=lambda k: t[k][0] - t[k][1])
print("  thinnest floating ice in the test run:")
for k in thin[:4]:
    print(f"    {k[0]/1e3:8.2f} {k[1]/1e3:6.2f}:  H test {t[k][0]-t[k][1]:7.1f}  (ref {r[k][0]-r[k][1]:7.1f})")
