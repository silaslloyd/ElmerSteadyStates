#!/usr/bin/env python3
"""
The time-step series (jobs/mech_dt_series.sh) against their reference: for every run, at every output time it shares
with the reference, how far its surface, grounding line, volume and flow pattern are from the reference's, and what
it cost. Stage 1 of the roadmap: does adapted FSSA at a large step match the small-step reference where full does not?

    python3 plots/mech_compare.py REF RUN [RUN ...] [--csv OUT.csv] [--plot OUT.png] [--all-times]

REF and RUN: job ids (the run directory work/*/runs/*_job<ID>_* is found) or run directories. Reads the boundary VTUs
(base_t*.pvtu, surface_t*.pvtu) only, so it works on runs still going (their steps so far).

Per run and time (only the last common time is printed unless --all-times):
  dZs rms / max     surface elevation difference (m), and where the max is
  dGL               grounded area difference / strip width (km): the mean grounding-line shift, + = run more advanced
  GL nodes          base nodes grounded in one run and floating in the other
  dV/V              ice volume difference (sum of nodal thickness), relative
  speed err         || |u_s| - |u_s,ref| ||_2 / || |u_s,ref| ||_2: the ice-stream pattern
  done              steps done / steps asked (a run that stopped early: blown up or still running)
  cost              solver wall time (s) per 1000 model years, from the Elmer log (Solver Timing)
"""
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_state_3d import vtu_steps  # noqa: E402
from vtu_frames import run_timing, surface  # noqa: E402


def find_run(a):
    if os.path.isdir(a):
        return a.rstrip("/")
    ds = sorted(glob.glob(f"work/*/runs/*_job{a}_*/"), key=os.path.getmtime)
    if not ds:
        raise SystemExit(f"{a}: no run directory work/*/runs/*_job{a}_* (run from the repo root, or give the directory)")
    return ds[-1].rstrip("/")


class Run:
    def __init__(self, arg):
        self.dir = find_run(arg)
        self.every, self.dt, offset = run_timing(self.dir)
        step_of = (lambda k: k * self.every) if offset else (lambda k: (k - 1) * self.every + 1)
        sif = open(os.path.join(self.dir, "run.sif")).read()
        self.mode = "adapted" if "[elmer] fssa = adapted" in sif else "full"
        m = re.search(r"Steady State Max Iterations\s*=\s*(?:Integer\s*)?(\d+)", sif)
        self.iters = int(m.group(1)) if m else 1
        m = re.search(r"_job(\d+)_", self.dir)
        self.job = m.group(1) if m else os.path.basename(self.dir)
        self.label = f"{self.mode} dt{self.dt:g}" + (f" x{self.iters}" if self.mode == "adapted" else "")
        self.files = {}
        for f in vtu_steps(self.dir):
            k = int(re.findall(r"_t(\d+)", os.path.basename(f))[0])
            self.files[round(step_of(k) * self.dt, 3)] = f
        # progress and cost from the Elmer log
        log = os.path.join(self.dir, "elmer.log")
        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        st = re.findall(r"MAIN: Time:\s*(\d+)\s*/\s*(\d+)", txt)
        self.done, self.asked = (int(st[-1][0]), int(st[-1][1])) if st else (0, 0)
        # "Solver time (CPU,REAL) for <solver>:  <cpu>  <real> (s)": the real (wall) times of every solver call
        secs = sum(float(s) for s in re.findall(r"Solver time \(CPU,REAL\) for [^\n]*?[-+0-9.eE]+\s+([-+0-9.eE]+)\s*\(s\)", txt))
        self.years = self.done * self.dt
        self.cost = 1000 * secs / self.years if self.years and secs else np.nan
        self.finished = "ALL DONE" in txt or (self.asked and self.done >= self.asked)
        self._cache = {}

    def state(self, t):
        if t not in self._cache:
            f = self.files[t]
            try:
                b, s = surface(f), surface(f.replace("base_t", "surface_t"))
            except Exception:
                b = s = None
            self._cache[t] = (b, s)
        return self._cache[t]


def key(d):
    return [(round(x), round(y)) for x, y in zip(d["x"], d["y"])]


def compare(r, ref, t):
    (b, s), (b0, s0) = r.state(t), ref.state(t)
    if b is None or s is None or b0 is None or s0 is None:
        return None
    k0 = {k: i for i, k in enumerate(key(s0))}
    ks = key(s)
    i = np.array([k0.get(k, -1) for k in ks])
    ok = i >= 0
    dz = s["z"][ok] - s0["z"][i[ok]]
    j = int(np.argmax(np.abs(dz)))
    xm, ym = s["x"][ok][j] / 1e3, s["y"][ok][j] / 1e3
    # thickness: surface minus base at the same (x, y) (the extruded mesh: the same columns)
    kb0 = {k: n for n, k in enumerate(key(b0))}
    kb = {k: n for n, k in enumerate(key(b))}

    def thick(sf, bs, kbm):
        ii = np.array([kbm.get(k, -1) for k in key(sf)])
        g = ii >= 0
        return np.sum(sf["z"][g] - bs["z"][ii[g]])
    V, V0 = thick(s, b, kb), thick(s0, b0, kb0)
    out = dict(rms=float(np.sqrt(np.mean(dz ** 2))), mx=float(dz[j]), xm=xm, ym=ym, dV=(V - V0) / V0)
    sp = np.sqrt(s["vx"] ** 2 + s["vy"] ** 2)[ok]
    sp0 = np.sqrt(s0["vx"] ** 2 + s0["vy"] ** 2)[i[ok]]
    out["speed"] = float(np.linalg.norm(sp - sp0) / np.linalg.norm(sp0))
    if "mask" in b and "mask" in b0:
        ib = np.array([kb0.get(k, -1) for k in key(b)])
        g = ib >= 0
        gr, gr0 = b["mask"][g] > -0.5, b0["mask"][ib[g]] > -0.5
        W = float(np.ptp(b0["y"]))
        area = float(np.ptp(b0["x"])) * W / len(b0["x"])           # mean nodal area (m^2)
        out["dGL"] = (gr.sum() - gr0.sum()) * area / W / 1e3
        out["glnodes"] = int(np.sum(gr != gr0))
    else:
        out["dGL"], out["glnodes"] = np.nan, 0
    return out


def main():
    args = sys.argv[1:]
    csv = png = None
    alltimes = "--all-times" in args
    args = [a for a in args if a != "--all-times"]
    for o in ("--csv", "--plot"):
        if o in args:
            k = args.index(o)
            v = args[k + 1]
            args = args[:k] + args[k + 2:]
            csv, png = (v, png) if o == "--csv" else (csv, v)
    if len(args) < 2:
        raise SystemExit(__doc__)
    ref = Run(args[0])
    runs = []
    for a in args[1:]:                       # a job not started yet (no run directory): skipped
        try:
            runs.append(Run(a))
        except SystemExit as e:
            print(f"  skipped: {e}")
    print(f"reference: job {ref.job} {ref.label}, {ref.done}/{ref.asked} steps, outputs at "
          f"{', '.join(f'{t:g}' for t in sorted(ref.files))} yr, cost {ref.cost:.0f} s / kyr")
    hdr = (f"{'job':>6} {'run':<18} {'t (yr)':>7} {'dZs rms':>8} {'dZs max':>8} {'at (km)':>14} {'dGL km':>7} "
           f"{'GL nodes':>8} {'dV/V':>9} {'speed err':>9} {'done':>9} {'s/kyr':>7}")
    print(hdr)
    rows = []
    for r in runs:
        common = sorted(set(r.files) & set(ref.files))
        if not common:
            print(f"{r.job:>6} {r.label:<18} no output times in common with the reference "
                  f"({r.done}/{r.asked} steps)")
            continue
        last = None
        for t in common:
            c = compare(r, ref, t)
            if c is None:
                continue
            last = (t, c)
            rows.append(dict(job=r.job, run=r.label, mode=r.mode, dt=r.dt, t=t, **c, done=r.done, asked=r.asked,
                             cost=r.cost))
            if alltimes:
                pr(r, t, c)
        if last and not alltimes:
            pr(r, *last)
    if csv and rows:
        import csv as C
        with open(csv, "w", newline="") as f:
            w = C.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"-> {csv}")
    if png and rows:
        plot(rows, ref, png)


def pr(r, t, c):
    print(f"{r.job:>6} {r.label:<18} {t:7.0f} {c['rms']:8.2f} {c['mx']:+8.1f} ({c['xm']:5.0f},{c['ym']:5.0f}) "
          f"{c['dGL']:+7.2f} {c['glnodes']:8d} {c['dV']:+9.2e} {c['speed']:9.3f} {r.done:4d}/{r.asked:<4d} "
          f"{r.cost:7.0f}" + ("" if r.finished else "  (not finished)"))


def plot(rows, ref, png):
    """accuracy (surface rms difference at the last common time) against cost, full and adapted"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    last = {}
    for row in rows:
        if row["job"] not in last or row["t"] > last[row["job"]]["t"]:
            last[row["job"]] = row
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, col in (("full", "#1f6fb4"), ("adapted", "#d1495b")):
        P = sorted((r for r in last.values() if r["mode"] == mode), key=lambda r: r["dt"])
        if not P:
            continue
        ax[0].plot([r["cost"] for r in P], [r["rms"] for r in P], "o-", color=col, label=mode)
        ax[1].plot([r["dt"] for r in P], [r["rms"] for r in P], "o-", color=col, label=mode)
        for r in P:
            ax[0].annotate(f"dt{r['dt']:g}" + ("" if r["done"] >= r["asked"] else " (stopped)"),
                           (r["cost"], r["rms"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax[0].axvline(ref.cost, color="0.5", ls=":", label=f"reference dt{ref.dt:g}")
    ax[0].set_xlabel("solver wall time per 1000 yr (s)"); ax[0].set_ylabel("surface rms difference from the reference (m)")
    ax[1].set_xlabel("time step (yr)")
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=0.3, which="both"); a.legend(frameon=False)
    fig.suptitle(f"against the dt{ref.dt:g} reference (job {ref.job}), at each run's last common output time")
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"-> {png}")


if __name__ == "__main__":
    main()
