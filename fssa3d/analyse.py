#!/usr/bin/env python3
"""Compare the fssa3d runs: the surface bump amplitude and roughness against time, and the end-state error against
the reference (smallest dt of the explicit / full runs, or --ref TAG).

    python3 fssa3d/analyse.py [--ref exp_full_dt1] [--runs DIR]
Prints a table, writes fssa3d/fssa3d.png.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "plots"))
from vtu_profile import read_vtu   # noqa: E402


def zs_of(path, H):
    v = read_vtu(path)
    p, f = v["points"], v["fields"]
    k = next(n for n in f if n.lower() == "zs")
    z = np.asarray(f[k]).reshape(len(p), -1)[:, 0]
    return p[:, 0], p[:, 1], z


def load(d, H):
    """[(step index, x, y, Zs)] of the surface VTUs of one run (serial: surf*.vtu)"""
    fs = sorted(f for f in glob.glob(os.path.join(d, "surf*.vtu")))
    out = []
    for i, f in enumerate(fs):
        try:
            x, y, z = zs_of(f, H)
        except Exception as e:   # a half-written or odd file
            print("  skip", f, e)
            continue
        out.append((i, x, y, z))
    return out


def rough(x, y, z):
    """rms second difference of Zs along x and y on the structured footprint (grid-scale noise)"""
    xs, ys = np.unique(np.round(x, 3)), np.unique(np.round(y, 3))
    g = np.full((len(ys), len(xs)), np.nan)
    ix, iy = np.searchsorted(xs, np.round(x, 3)), np.searchsorted(ys, np.round(y, 3))
    g[iy, ix] = z
    d2 = np.concatenate([np.diff(g, 2, axis=1).ravel(), np.diff(g, 2, axis=0).ravel()])
    return float(np.sqrt(np.nanmean(d2 ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--ref", default=None)
    ap.add_argument("--H", type=float, default=1000.0)
    ap.add_argument("--out", default=os.path.join(HERE, "fssa3d.png"))
    a = ap.parse_args()
    tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.runs, "*")) if os.path.isdir(d))
    runs = {t: load(os.path.join(a.runs, t), a.H) for t in tags}
    runs = {t: r for t, r in runs.items() if r}
    if not runs:
        sys.exit("no surface VTUs found under " + a.runs)
    ref = a.ref or min((t for t in runs if t.startswith("exp_full")), default=None,
                       key=lambda t: float(t.split("dt")[-1]) if "dt" in t else 1e9) or sorted(runs)[0]
    print("reference:", ref)
    rx, ry, rz = runs[ref][-1][1:]
    key = lambda x, y: (np.round(x, 3), np.round(y, 3))
    order_r = np.lexsort(key(rx, ry))
    print(f"{'run':24s} {'steps':>5s} {'max Zs-H (m) first':>20s} {'last':>9s} {'rough (m)':>10s} {'rms vs ref (m)':>15s} {'log':>6s}")
    hist = {}
    for t, r in runs.items():
        amp = [float(np.nanmax(np.abs(z - a.H))) for _, _, _, z in r]
        hist[t] = amp
        i, x, y, z = r[-1]
        o = np.lexsort(key(x, y))
        err = float(np.sqrt(np.mean((z[o] - rz[order_r]) ** 2))) if len(z) == len(rz) else float("nan")
        log = os.path.join(a.runs, t, "elmer.log")
        ok = "ok" if os.path.exists(log) and "ELMER SOLVER FINISHED" in open(log, errors="ignore").read() else "NOT FIN"
        print(f"{t:24s} {len(r):5d} {amp[0]:20.2f} {amp[-1]:9.2f} {rough(x, y, z):10.3f} {err:15.3f} {ok:>6s}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for t, h in sorted(hist.items()):
            ax.semilogy(range(len(h)), np.maximum(h, 1e-6), "-o" if t == ref else "-", ms=3, label=t)
        ax.set_xlabel("output index")
        ax.set_ylabel("max |Zs - H| (m)")
        ax.legend(fontsize=6, ncol=2)
        fig.savefig(a.out, dpi=150, bbox_inches="tight")
        print("->", a.out)
    except Exception as e:
        print("no plot:", e)


if __name__ == "__main__":
    main()
