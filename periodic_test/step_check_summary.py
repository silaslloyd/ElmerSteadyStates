#!/usr/bin/env python3
"""Summary of the per-step check (STEP_CHECK=1, elmer/StepCheck.F90) over the Elmer runs of a job: one line per run
(chunk) with the largest and the median per-step value of the surface CFL, the floating-base CFL, |dZs|, |dZb| and
|dZb|/H, where the largest was, and the relaxation number R; --steps prints every step.

    python3 periodic_test/step_check_summary.py RUN_DIR [RUN_DIR ...]  [--steps]
    python3 periodic_test/step_check_summary.py --last 5 [WORK_DIR]       (the 5 newest run dirs, default work dir
                                                                          work/test_periodic_gl_50km_tri)"""
import argparse, glob, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="*")
ap.add_argument("--last", type=int)
ap.add_argument("--steps", action="store_true")
a = ap.parse_args()
dirs = a.dirs
if a.last:
    w = dirs[0] if dirs else "work/test_periodic_gl_50km_tri"
    dirs = sorted(glob.glob(os.path.join(w, "runs", "*/")), key=os.path.getmtime)[-a.last:]
num = r"([-+0-9.EeNa*]+)"
r1 = re.compile(r"step check t\s*" + num + r": CFL surface\s*" + num + r" \(x\s*" + num + r" y\s*" + num
                + r" km;\s*(\d+) nodes > 1\), floating base\s*" + num + r" \(x\s*" + num + r" y\s*" + num + r" km;\s*(\d+)")
r2 = re.compile(r"step check t\s*" + num + r": max \|dZs\|\s*" + num + r" m \(x\s*" + num + r" y\s*" + num
                + r"\), max \|dZb\| floating\s*" + num + r" m \(x\s*" + num + r" y\s*" + num + r"\), max \|dZb\|/H\s*"
                + num + r" \(x\s*" + num + r" y\s*" + num + r"\), relaxation number R\s*" + num)
f = lambda s: float(s) if re.fullmatch(r"[-+0-9.Ee]+", s) else np.nan
for d in dirs:
    log = os.path.join(d, "elmer.log")
    if not os.path.exists(log):
        continue
    A, B = [], []
    for line in open(log, errors="replace"):
        m = r1.search(line)
        if m:
            A.append([f(g) for g in m.groups()])
        m = r2.search(line)
        if m:
            B.append([f(g) for g in m.groups()])
    if not A:
        print(f"{d}: no step check lines (STEP_CHECK=1?)")
        continue
    A, B = np.array(A), np.array(B) if B else np.full((len(A), 11), np.nan)
    if a.steps:
        for x, y in zip(A, B):
            print(f"  t {x[0]:8.2f}: CFL surface {x[1]:7.3f} at x {x[2]:7.2f} y {x[3]:6.2f} ({x[4]:.0f} nodes > 1), "
                  f"floating base {x[5]:7.3f} ({x[8]:.0f} > 1); |dZs| {y[1]:8.3f} m at x {y[2]:7.2f}, "
                  f"|dZb| {y[4]:8.3f} m, |dZb|/H {y[7]:.2e}")
    k = lambda c: int(np.nanargmax(c)) if np.isfinite(c).any() else 0
    i1, i5, j1, j4, j7 = k(A[:, 1]), k(A[:, 5]), k(B[:, 1]), k(B[:, 4]), k(B[:, 7])
    print(f"{os.path.basename(d.rstrip('/'))}: {len(A)} steps")
    print(f"  CFL surface   max {A[i1, 1]:7.3f} (t {A[i1, 0]:g}, x {A[i1, 2]:.2f} y {A[i1, 3]:.2f} km), median of the "
          f"step maxima {np.nanmedian(A[:, 1]):.3f}, steps with nodes > 1: {int((A[:, 4] > 0).sum())}")
    print(f"  CFL fl. base  max {A[i5, 5]:7.3f} (t {A[i5, 0]:g}, x {A[i5, 6]:.2f} y {A[i5, 7]:.2f} km), median "
          f"{np.nanmedian(A[:, 5]):.3f}, steps with nodes > 1: {int((A[:, 8] > 0).sum())}")
    print(f"  |dZs| per step max {B[j1, 1]:.3f} m (t {B[j1, 0]:g}, x {B[j1, 2]:.2f} y {B[j1, 3]:.2f}), median "
          f"{np.nanmedian(B[:, 1]):.3f}; |dZb| floating max {B[j4, 4]:.3f} m (x {B[j4, 5]:.2f} y {B[j4, 6]:.2f}); "
          f"|dZb|/H max {B[j7, 7]:.2e}")
    print(f"  relaxation number R (explicit stable < 2): {np.nanmax(B[:, 10]):.3g}")
