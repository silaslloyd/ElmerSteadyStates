#!/usr/bin/env python3
"""plot_sweep.py

The temperate-patch sweep so far (jobs/gl_sweep.cmd): every segment's sweep.csv (and bisect.csv) of a sweep folder in one figure,
against the patch length d (km): the grounding line and the head of the patch, Newton iterations (and the step), the cold bed's
warmest temperature and the number of cold nodes above melting, the water sheet (h_min, q at the grounding line); the points
that are NOT consistent (cold bed above melting or the water sheet dry in the patch) are marked, the bisection points as open
markers.

    python3 diag/plot_sweep.py                                   # work/gl_ssa_gl549/sweep_gl549s100
    python3 diag/plot_sweep.py work/gl_ssa_gl549/sweep_gl549s100 --png sweep.png
    python3 diag/plot_sweep.py --start work/gl_ssa_gl549/branch_gl549s100/steady_temperate.npz

The FULLY TEMPERATE start (d = the whole grounded bed, the Newton's steady_temperate.npz) is marked as a black diamond and the
first line of the table: --start STATE (default: the branch state beside the sweep folder, work/<case>/branch_<tag>/steady_temperate.npz,
when it exists; --no-start to leave it out). It has no cold bed, so only its grounding line and head (d = grounding line) are shown.
"""
import argparse
import csv
import glob
import os
import re
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
    ap.add_argument("--start", help="the fully temperate steady state (default: found beside the sweep folder)")
    ap.add_argument("--no-start", action="store_true")
    a = ap.parse_args()
    start = None
    if not a.no_start:
        path = a.start
        if not path:
            m = re.match(r"(.*/)sweep_(.+?)/*$", a.folder.rstrip("/") + "/")
            cand = os.path.join(m.group(1), f"branch_{m.group(2)}", "steady_temperate.npz") if m else ""
            path = cand if os.path.exists(cand) else None
        if path:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from ess.spectrum import gl_position
                from ess.statefile import load_state
                s = load_state(path)
                start = gl_position(s.frozen, s.diag) / 1e3
            except Exception as ex:                                  # noqa: BLE001
                print(f"start state {path}: not read ({ex!r})")
    rows, brows = read(a.folder), read_b(a.folder)
    if not rows:
        print(f"no sweep.csv rows in {a.folder}/seg_*/ yet (the first point has not converged)")
        if start is None:
            sys.exit(1)
    ok = np.array([str(r.get("converged")) == "1" for r in rows])
    print(f"{ok.sum()} converged of {len(rows)} rows, {len(brows)} bisection rows")
    hdr = f"{'d km':>8} {'GL km':>8} {'head km':>8} {'its':>4} {'step':>6} {'cold>Tm':>8} {'hydro':>8} {'consist':>8}"
    print(hdr)
    if start is not None:
        print(f"{start:8.2f} {start:8.2f} {0.0:8.2f} {'-':>4} {'-':>6} {'0':>8} {'-':>8} {'(start)':>8}  fully temperate")
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
    if start is not None:
        for k, v in ((0, start), (1, 0.0)):
            ax[k].plot([start], [v], "D", color="k", ms=6, label="fully temperate start" if k == 0 else None)
        ax[3].plot([start], [273.15], "D", color="k", ms=6)
    ax[3].axhline(273.15, color="0.5", ls=":", lw=0.8)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(a.png)
    print(f"plot: {a.png}")


if __name__ == "__main__":
    main()
