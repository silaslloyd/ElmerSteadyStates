#!/usr/bin/env python3
"""The grounding line, surface and basal thermal state over ALL the output steps of an Elmer VTU directory (e.g. an
ElmerStreams flowline run): is it at a steady state, and where is it going?

Per output step (.pvtu, or groups of .vtu by step number; --every N: every Nth): the model time (Elmer's TimeValue
field data, or a .pvd collection's timestep, when there), the grounding line (the largest x of a grounded column: the
ice base within --tol m of the "bedrock" field, else the GroundedMask field >= 0 at the base), the divide surface,
the ice volume per unit width (mean across y), and at the base: how much of the grounded bed is temperate (T within
0.01 K of 273.15) and the basal temperature just upstream of the grounding line. Also prints the .sif files found in
the directory (time stepping, output intervals, the friction coefficient line) for reference. Writes DIR's
name + _history.csv next to plots/data/.

    python3 plots/vtu_history.py DIR [--every 1] [--tol 1.0] [--out FILE.csv]
"""
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vtu_profile import pieces_of, read_vtu, step_of, field  # noqa: E402


def steps(dirn):
    pv = sorted(set(glob.glob(os.path.join(dirn, "**", "*.pvtu"), recursive=True)), key=lambda p: (step_of(p), p))
    if pv:
        return [(step_of(p), pieces_of(p)) for p in pv]
    vt = sorted(set(glob.glob(os.path.join(dirn, "**", "*.vtu"), recursive=True)))
    by = {}
    for p in vt:
        by.setdefault(step_of(p), []).append(p)
    return sorted(by.items())


def pvd_times(dirn):
    t = {}
    for f in glob.glob(os.path.join(dirn, "**", "*.pvd"), recursive=True):
        for ds in ET.parse(f).getroot().iter("DataSet"):
            t[os.path.basename(ds.get("file", ""))] = float(ds.get("timestep", "nan"))
    return t


def main():
    a = sys.argv[1:]

    def opt(name, conv=str, default=None):
        if name not in a:
            return default
        i = a.index(name)
        v = conv(a[i + 1])
        del a[i:i + 2]
        return v
    every = opt("--every", int, 1)
    tol = opt("--tol", float, 1.0)
    out = opt("--out")
    if not a:
        raise SystemExit(__doc__)
    dirn = a[0]
    for f in sorted(glob.glob(os.path.join(dirn, "**", "*.sif"), recursive=True))[:5]:
        print(f"== {f}")
        for ln in open(f, errors="replace"):
            if re.search(r"Timestep (Sizes|Intervals)|Output Intervals?|Weertman Friction|TDepSliding|Restart File",
                         ln) and not ln.lstrip().startswith("!"):
                print("   " + ln.rstrip())
    st = steps(dirn)
    print(f"{len(st)} output steps under {dirn}")
    pvd = pvd_times(dirn)
    rows = []
    for k, (s, files) in enumerate(st):
        if k % every and k != len(st) - 1:
            continue
        P, F, meta = [], {}, {}
        for p in files:
            r = read_vtu(p)
            P.append(r["points"])
            meta.update(r["meta"])
            for kk, v in r["fields"].items():
                F.setdefault(kk, []).append(v)
        P = np.concatenate(P)
        F = {kk: np.concatenate(v) for kk, v in F.items() if sum(len(x) for x in v) == len(P)}
        t = float(meta["TimeValue"][0]) if "TimeValue" in meta else pvd.get(os.path.basename(files[0]), np.nan)
        x, y, z = P[:, 0] / 1e3, P[:, 1] / 1e3, P[:, 2]
        col = np.round(x * 1e3).astype(np.int64) * 10 ** 8 + np.round(y * 1e3).astype(np.int64)
        uc, ci = np.unique(col, return_inverse=True)
        cx = (uc // 10 ** 8) / 1e3
        top = np.full(len(uc), -np.inf)
        np.maximum.at(top, ci, z)
        kb, kbed = field(F, "zb"), field(F, "bedrock")
        kT, kgm = field(F, "temperature"), field(F, "groundedmask")
        # the ice base of each column: the Zb field's nonzero nodes (Elmer writes 0 where undefined)
        def colmean(v, m):
            c = np.bincount(ci[m], minlength=len(uc))
            return np.where(c > 0, np.bincount(ci[m], weights=v[m], minlength=len(uc)) / np.maximum(c, 1), np.nan)
        zb = colmean(F[kb], F[kb] != 0) if kb else np.full(len(uc), np.nan)
        if kbed:
            bed = colmean(F[kbed], F[kbed] != 0)
            grounded = zb - bed < tol
            how = f"Zb - bedrock < {tol:g} m"
        elif kgm:
            near = np.abs(z - np.nan_to_num(zb[ci], nan=1e9)) < 1.0
            grounded = colmean(F[kgm], near) > -0.5
            how = "GroundedMask at the base"
        else:
            raise SystemExit("no bedrock or GroundedMask field: cannot place the grounding line")
        xgl = cx[grounded].max() if grounded.any() else np.nan
        # basal temperature: the node of each column nearest the ice base
        Tb = np.full(len(uc), np.nan)
        if kT and kb:
            d = np.abs(z - zb[ci])
            best = np.full(len(uc), np.inf)
            np.minimum.at(best, ci, d)
            sel = d <= best[ci] + 1e-6
            Tb = colmean(F[kT], sel)
        gx = grounded & np.isfinite(Tb)
        temperate = np.mean(Tb[gx] > 273.15 - 0.01) if gx.any() else np.nan
        up = gx & (cx > xgl - 20) & (cx <= xgl)
        tb_gl = np.nanmean(Tb[up]) - 273.15 if up.any() else np.nan
        # the volume per unit width: (top - base) along x, mean across y
        thick = top - np.where(np.isfinite(zb), zb, np.nan)
        xs, xi = np.unique(np.round(cx, 3), return_inverse=True)
        hm = np.bincount(xi, weights=np.nan_to_num(thick)) / np.bincount(xi)
        vol = np.trapezoid(hm, xs * 1e3) if hasattr(np, "trapezoid") else np.trapz(hm, xs * 1e3)
        zdiv = top[cx == cx.min()].mean()
        rows.append((s, t, xgl, zdiv, vol / 1e6, temperate, tb_gl))
        print(f"step {s:6d}  t {t:10.1f} yr  GL {xgl:8.2f} km  divide Zs {zdiv:7.1f} m  volume {vol / 1e6:9.3f} km^2  "
              f"temperate bed {temperate:5.1%}  T_b(GL-20..GL) {tb_gl:+6.2f} C   [{how}]", flush=True)
    out = out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                              re.sub(r"\W+", "_", os.path.basename(os.path.normpath(dirn))) + "_history.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("step,time_yr,x_gl_km,zs_divide_m,volume_km2,temperate_fraction,Tb_near_gl_C\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
