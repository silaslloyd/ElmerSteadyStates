#!/usr/bin/env python3
"""plot_sweep.py

The temperate-patch sweep so far (jobs/gl_sweep.cmd): every segment's sweep.csv (and bisect.csv) of a sweep folder in one figure,
against the patch length d (km): the grounding line and the head of the patch, Newton iterations (and the step), the cold bed's
warmest temperature and the number of cold nodes above melting, the water sheet (h_min, q at the grounding line); the points
that are NOT consistent (cold bed above melting or the water sheet dry in the patch) are marked, the bisection points as open
markers.

    python3 diag/plot_sweep.py                                   # work/gl_ssa_gl549/sweep_gl549s100
    python3 diag/plot_sweep.py work/gl_ssa_gl549/sweep_gl549s100 --png sweep.png
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np


def read(path):
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "seg_*", "sweep.csv"))):
        for r in csv.DictReader(open(f)):
            r["seg"] = os.path.basename(os.path.dirname(f))
            rows.append(r)
    return rows


def read_b(path):
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "seg_*", "bisect.csv"))):
        rows += list(csv.DictReader(open(f)))
    return rows


def col(rows, k):
    out = []
    for r in rows:
        try:
            out.append(float(r[k]))
        except (KeyError, ValueError, TypeError):
            out.append(np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="work/gl_ssa_gl549/sweep_gl549s100")
    ap.add_argument("--png", default="sweep.png")
    a = ap.parse_args()
    rows, brows = read(a.folder), read_b(a.folder)
    if not rows:
        sys.exit(f"no sweep.csv rows in {a.folder}/seg_*/ yet (the first point has not converged)")
    ok = np.array([str(r.get("converged")) == "1" for r in rows])
    print(f"{ok.sum()} converged of {len(rows)} rows, {len(brows)} bisection rows")
    hdr = f"{'d km':>8} {'GL km':>8} {'head km':>8} {'its':>4} {'step':>6} {'cold>Tm':>8} {'hydro':>8} {'consist':>8}"
    print(hdr)
    for r in rows:
        print(f"{float(r['d_km']):8.2f} {float(r['x_gl_km']):8.2f} {float(r['x_head_km']):8.2f} {r['newton_its']:>4} "
              f"{float(r.get('step_km') or 0):6.1f} {r.get('n_cold_above_Tm', ''):>8} {r.get('hydro_status', ''):>8} "
              f"{r.get('consistent', ''):>8}  {'' if str(r['converged']) == '1' else 'FAILED'} {r.get('note', '')}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), dpi=120)
    ax = ax.ravel()
    for name, rr in (("sweep", rows), ("bisect", brows)):
        if not rr:
            continue
        d = col(rr, "d_km")
        conv = np.array([str(r.get("converged")) == "1" for r in rr])
        cons = col(rr, "consistent")
        fc = "none" if name == "bisect" else None
        for k, (key, lab) in enumerate([("x_gl_km", "grounding line (km)"), ("x_head_km", "head of the patch (km)"),
                                        ("newton_its", "Newton iterations"), ("Tb_cold_max", "warmest cold bed T (K)"),
                                        ("h_min", "water sheet h_min (m)"), ("q_gl", "water flux at the GL")]):
            v = col(rr, key)
            m = conv & np.isfinite(v)
            good, bad = m & (cons != 0), m & (cons == 0)
            ax[k].plot(d[m], v[m], "-" if name == "sweep" else "none", color="0.6", lw=0.8)
            ax[k].plot(d[good], v[good], "o", mfc=fc or "C0", mec="C0", ms=5, label=f"{name}: consistent")
            ax[k].plot(d[bad], v[bad], "s", mfc=fc or "C3", mec="C3", ms=5, label=f"{name}: not consistent")
            ax[k].set_xlabel("patch length d (km)")
            ax[k].set_ylabel(lab)
            ax[k].invert_xaxis()
    ax[3].axhline(273.15, color="0.5", ls=":", lw=0.8)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(a.png)
    print(f"plot: {a.png}")


if __name__ == "__main__":
    main()
