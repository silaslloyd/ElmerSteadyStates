#!/usr/bin/env python3
"""track_gl.py

Tracks the grounding line of the transient branches (jobs/ssa_start.cmd, jobs/ssa_start_gl.cmd) from their slurm logs: for each
branch the chunks so far, the grounding line, its retreat rate over several windows, an exponential fit
x(k) = x* + A exp(-k / tau) (k in chunks of 1000 yr) with the position it heads for and the time left, and the wall time per chunk.
Across all branches, the pooled fit of  rate = (x - x*) / tau  (retreat rate against position, every 5-chunk window, the first
--skip chunks of each branch left out: that is the model adjusting to the start state, not the drift). Branches started on
both sides of the steady position make that fit well conditioned: the rate changes sign at x*.

    python3 diag/track_gl.py                          # slurms/ssa_start_*.out and slurms/ssa_start_gl_*.out
    python3 diag/track_gl.py slurms/ssa_start_32411.out slurms/ssa_start_gl_*.out --png track_gl.png
options: --skip 5  --window 5  --fit-last 40 (chunks used by the per-branch exponential fit)  --png FILE

A branch is named by its log's "branch_<tag>/" path (several logs of one branch, a resubmitted job, are joined in job order).
CHUNKS ARE COUNTED FROM THE "run: chunk i/N t = ... rms change ..." LINES (one per chunk of 1000 yr). The line
"run: contact grounding line A -> B km" is printed only by a chunk in which the grounding line moved; a chunk without it
did not move (counting the contact lines instead undercounts the chunks, and so overstates the rate per chunk).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

LINE = re.compile(r"^(\d\d):(\d\d):(\d\d) run: contact grounding line ([\d.]+) -> ([\d.]+) km")
CHUNK = re.compile(r"^(\d\d):(\d\d):(\d\d) run: chunk \d+/\d+\s+t = \S+\s+rms change ([\d.eE+-]+)")
BRANCH = re.compile(r"branch_([A-Za-z0-9_.-]+)/")


def jobid(path):
    m = re.search(r"_(\d+)\.out$", path)
    return int(m.group(1)) if m else 0


def read_logs(paths):
    """{tag: dict(x0, x (position after each chunk), sec (wall second of each chunk, unwrapped), rms, files)}"""
    runs = {}
    for p in sorted(paths, key=jobid):
        tag, ev = None, []
        for ln in open(p, errors="replace"):
            if tag is None:
                m = BRANCH.search(ln)
                if m and "states every" in ln:
                    tag = m.group(1)
            m = CHUNK.match(ln)
            if m:
                ev.append(("chunk", int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)), float(m.group(4))))
                continue
            m = LINE.match(ln)
            if m:
                ev.append(("gl", float(m.group(4)), float(m.group(5))))
        if not any(e[0] == "gl" for e in ev):
            continue
        tag = tag or os.path.basename(p)
        r = runs.setdefault(tag, dict(x0=None, x=[], sec=[], rms=[], files=[]))
        cur = r["x"][-1] if r["x"] else None                  # the position now (None until the first line gives it)
        if r["x0"] is None:
            r["x0"] = next(e[1] for e in ev if e[0] == "gl")
            cur = r["x0"]
        last = (r["sec"][-1] % 86400) if r["sec"] else None
        base = (r["sec"][-1] - last) if r["sec"] else 0
        for e in ev:
            if e[0] == "chunk":
                s = e[1]
                if last is not None and s < last - 3600:         # past midnight
                    base += 86400
                last = s
                r["sec"].append(base + s)
                r["x"].append(cur)
                r["rms"].append(e[2])
            else:                                                # the contact line follows its chunk line
                cur = e[2]
                if r["x"]:
                    r["x"][-1] = cur
        r["files"].append(p)
    for r in runs.values():
        r["x"], r["sec"] = np.array(r["x"], float), np.array(r["sec"], float)
    return runs


def rate(x, k0, k1):
    """retreat rate (km per chunk) between chunk indices k0 < k1 (positive = retreating)"""
    return (x[k0] - x[k1]) / (k1 - k0)


def fit_exp(k, x):
    """x* + A exp(-k/tau) -> ((x*, A, tau), errors) or None"""
    try:
        from scipy.optimize import curve_fit
        f = lambda kk, xs, A, tau: xs + A * np.exp(-kk / tau)
        A0 = x[0] - x[-1]
        p, c = curve_fit(f, k, x, p0=[x[-1] - A0, A0 if A0 else 1.0, 60.0], maxfev=20000,
                         bounds=([0, -400, 3.0], [2000, 400, 2000.0]))
        return p, np.sqrt(np.diag(c))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--skip", type=int, default=5, help="chunks left out at the start of each branch (the adjustment)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--fit-last", type=int, default=40)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    paths = a.logs or sorted(glob.glob("slurms/ssa_start_*.out"))
    if not paths:
        sys.exit("no logs (slurms/ssa_start_*.out)")
    runs = read_logs(paths)
    if not runs:
        sys.exit("no 'contact grounding line' lines in " + " ".join(paths))

    pool_x, pool_r, pool_tag = [], [], []
    for tag, r in runs.items():
        x, n = r["x"], len(r["x"])
        nstill = int(np.sum(np.diff(np.concatenate([[r["x0"]], x])) == 0))
        print(f"\n== {tag}: {n} chunks ({n} kyr), grounding line {r['x0']:.2f} -> {x[-1]:.2f} km, "
              f"{nstill} chunks without a move   [{', '.join(os.path.basename(f) for f in r['files'])}]")
        if r["rms"]:
            print(f"   last rms change {r['rms'][-1]:.2e}")
        w = a.window
        if n > a.skip + w:
            xs = np.concatenate([[r["x0"]], x])                   # xs[k] = after k chunks
            print("   retreat rate, km per 1000 yr (positive = retreating):")
            wins = [("last 5", 5), ("last 10", 10), ("last 20", 20), ("since skip", n - a.skip)]
            print("     " + "   ".join(f"{name} {rate(xs, n - m, n):+.3f}" for name, m in wins if 0 < m <= n))
            for k0 in range(a.skip, n - w + 1, w):
                pool_x.append(0.5 * (xs[k0] + xs[k0 + w]))
                pool_r.append(rate(xs, k0, k0 + w))
                pool_tag.append(tag)
            kk = np.arange(a.skip, n + 1)[-a.fit_last:]
            fit = fit_exp(kk - kk[0], xs[kk]) if len(kk) >= 12 else None
            if fit and (fit[1][0] > 25.0 or fit[1][2] > fit[0][2]):
                print("   exponential fit: not constrained (the grounding line has hardly moved, or the rate is flat)")
            elif fit:
                (xst, A, tau), (exs, eA, etau) = fit
                left = abs(xs[-1] - xst)
                t1 = tau * np.log(left / 1.0) if left > 1.0 else 0.0
                print(f"   exponential fit, last {len(kk)} chunks: x* = {xst:.1f} +- {exs:.1f} km, tau = {tau:.0f} +- {etau:.0f} kyr; "
                      f"{xs[-1] - xst:+.1f} km to go, {t1:.0f} kyr to within 1 km")
            else:
                print(f"   exponential fit: not enough chunks after the first {a.skip} (or no convergence)")
        if n > 3:
            dt = np.diff(r["sec"])
            dt = dt[(dt > 0) & (dt < 4 * 3600)]
            if len(dt):
                m = float(np.median(dt[-20:])) / 60
                print(f"   wall time per chunk (median of the last {min(len(dt), 20)}): {m:.1f} min -> 100 kyr = {100 * m / 60:.0f} h")

    print("\n== pooled over the branches: rate = (x - x*) / tau   (retreat rate against position)")
    px, pr = np.array(pool_x), np.array(pool_r)
    spread = (px.max() - px.min()) if len(px) else 0.0
    if len(px) >= 6 and spread > 3.0:
        (s, c), cov = np.polyfit(px, pr, 1, cov=True)
        xstar = -c / s
        tau = 1.0 / s if s > 0 else float("nan")
        es, ec = np.sqrt(np.diag(cov))
        exs = abs(xstar) * np.sqrt((ec / c) ** 2 + (es / s) ** 2 - 2 * cov[0, 1] / (s * c)) if s and c else float("nan")
        print(f"   {len(px)} windows over {px.min():.1f}-{px.max():.1f} km: x* = {xstar:.1f} +- {exs:.1f} km, tau = {tau:.0f} kyr "
              f"(slope {s:.2e} per km per kyr)")
        if s <= 0:
            print("   slope not positive: the rate does not fall as the grounding line retreats (no restoring drift found yet)")
        res = pr - np.polyval([s, c], px)
        print(f"   scatter of the windows about the line: {res.std():.3f} km per 1000 yr")
        by = {}
        for t, x_, r_ in zip(pool_tag, px, pr):
            by.setdefault(t, []).append((x_, r_))
        for t, v in by.items():
            v = np.array(v)
            print(f"   {t}: {len(v)} windows, mean position {v[:, 0].mean():.1f} km, mean rate {v[:, 1].mean():+.3f}")
    else:
        print(f"   needs >= 6 windows over more than 3 km of position (have {len(px)}, spread {spread:.1f} km)")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
        for tag, r in runs.items():
            n = len(r["x"])
            ax[0].plot(np.arange(0, n + 1), np.concatenate([[r["x0"]], r["x"]]), label=tag)
        ax[0].set_xlabel("chunks (kyr)")
        ax[0].set_ylabel("contact grounding line (km)")
        ax[0].legend(frameon=False)
        for tag in runs:
            m = [i for i, t in enumerate(pool_tag) if t == tag]
            if m:
                ax[1].plot(px[m], pr[m], ".", label=tag)
        if len(px) >= 6 and spread > 3.0:
            xx = np.linspace(px.min(), px.max(), 20)
            ax[1].plot(xx, np.polyval([s, c], xx), "k-", lw=1)
        ax[1].axhline(0, color="0.6", lw=0.8, ls=":")
        ax[1].set_xlabel("grounding line (km)")
        ax[1].set_ylabel("retreat rate (km per 1000 yr)")
        fig.tight_layout()
        fig.savefig(a.png)
        print(f"plot: {a.png}")


if __name__ == "__main__":
    main()
