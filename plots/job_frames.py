#!/usr/bin/env python3
"""All the outputs of one job, across its remeshes, or of a run made of several jobs (restarts): every run directory
of the job(s) (runs/*_job<JOB>_*: one per mesh with `ess run --remesh-auto`) packed with plots/vtu_frames.py into
OUTBASE_seg001.npz, OUTBASE_seg002.npz, .. in time order, each with its true model time. plots/plot_state_3d.py takes
them all at once (one page, the mesh changing where the run remeshed):

    python3 plots/job_frames.py WORKDIR JOB[,JOB2,..] OUTBASE [--every N | --every-yr Y] [--t0 T0] [vtu_frames options]
    python3 plots/plot_state_3d.py OUTBASE_seg*.npz ...

Times: each run directory's start time is read from its parameters.lua (ess_sl_off / ess_acc_off: the model time the
run started at, with --sea-level / --accumulation, also after restarts with TIME_OFFSET); without them the times the
earlier segments ran (their logs' last "MAIN: Time") plus T0 (--t0, default 0).
Several jobs (a run restarted from a checkpoint): in the order given; a job's frames after the next job's start are
left out (the restart replaced them).
--every N: every N-th regular output, counted across all segments (and the very last); each segment's first step on
its new mesh (basestart, see ess/elmer.py) is always kept. Default 1: every output. --every-yr Y instead: the
outputs at multiples of Y yr of model time (e.g. 1000: 1000, 2000, ..), whatever the segments.
The new segments are written under temporary names and replace the old OUTBASE_seg*.npz (all of them, also another
job's) only at the end, so a glob never mixes sets and a download during packing still gets the old, complete set.
"""
import glob
import os
import re
import subprocess
import sys


def start_time(d):
    """the model time a run directory started at (its parameters.lua's ess_sl_off / ess_acc_off), or None"""
    try:
        lua = open(os.path.join(d, "parameters.lua")).read()
    except OSError:
        return None
    m = re.search(r"^ess_(?:sl|acc)_off\s*=\s*(\S+)", lua, re.M)
    return float(m.group(1)) if m else None


def run_length(d):
    """the model time a run directory covered (its log's last MAIN: Time)"""
    tend = 0.0
    log = os.path.join(d, "elmer.log")
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            m = re.match(r"\s*MAIN: Time: \d+/\d+:\s+(\S+)", line)
            if m:
                tend = float(m.group(1))
    return tend


def main():
    a = sys.argv[1:]
    if len(a) < 3:
        raise SystemExit(__doc__)
    work, jobs, outbase = a[0], [j for j in a[1].split(",") if j], a[2]
    rest = a[3:]
    t0 = 0.0
    if "--t0" in rest:
        k = rest.index("--t0")
        t0 = float(rest[k + 1])
        del rest[k:k + 2]
    every_yr = None                                # --every-yr Y: the outputs at multiples of Y yr (model time)
    if "--every-yr" in rest:
        k = rest.index("--every-yr")
        every_yr = float(rest[k + 1])
        del rest[k:k + 2]
    every = 1
    if "--every" in rest:
        k = rest.index("--every")
        every = max(1, int(rest[k + 1]))
        del rest[k:k + 2]
    segs = []                                      # (job index, run dir, start time, length)
    off = t0
    for ji, job in enumerate(jobs):
        dirs = sorted(glob.glob(os.path.join(work, "runs", f"*_job{job}_*")))
        if not dirs:
            raise SystemExit(f"no run directories of job {job} in {work}/runs")
        for d in dirs:
            ts = start_time(d)
            off = ts if ts is not None else off
            ln = run_length(d)
            segs.append((ji, d, off, ln))
            off += ln
    # a job's segments end where the next job starts (a restart from a checkpoint replaced the rest)
    cut = {}
    for ji in range(len(jobs) - 1):
        nxt = [s for s in segs if s[0] == ji + 1]
        if nxt:
            cut[ji] = nxt[0][2]
    here = os.path.dirname(os.path.abspath(__file__))
    counts = [len(glob.glob(os.path.join(d, "base_t*.pvtu"))) for _, d, _, _ in segs]
    total = sum(counts)
    g0, made = 0, []
    for k, ((ji, d, ts, ln), nv) in enumerate(zip(segs, counts), 1):
        if every_yr:                               # by model time: output n at ts + its step x dt
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from vtu_frames import output_steps, run_timing
            ev, dt_, of_ = run_timing(d)
            sched = output_steps(d)
            stp = (lambda q: sched[q - 1]) if sched else (lambda q: of_ + 1 + (q - 1) * ev)
            tt = [ts + stp(n + 1) * dt_ for n in range(nv)]
            keep = [n for n in range(nv) if abs(tt[n] / every_yr - round(tt[n] / every_yr)) < 1e-6
                    or g0 + n == total - 1]
        else:
            keep = [n for n in range(nv) if (g0 + n) % every == 0 or g0 + n == total - 1]
        g0 += nv
        tmax = cut.get(ji)
        if tmax is not None and ts >= tmax - 1e-9:
            print(f"segment {k}: {os.path.basename(d)}: t {ts:g} yr, after job {jobs[ji + 1]}'s start: left out")
            continue
        print(f"segment {k}: {os.path.basename(d)}: {nv} outputs ({len(keep)} kept), t {ts:g} .. {ts + ln:g} yr")
        has_start = bool(glob.glob(os.path.join(d, "basestart_t*.pvtu")))   # the first step on its mesh (always packed)
        if keep or has_start:
            out = f"{outbase}_seg{len(made) + 1:03d}.npz.part.npz"   # renamed when all are done (a download
                                                                      # meanwhile still finds the old set)
            # up to and including the next job's start: that job continues from this output (its own first frame
            # is a step later), so the output at the restart time is this job's
            extra = ["--tmax", repr(tmax + 1e-6)] if tmax is not None else []
            r = subprocess.run([sys.executable, os.path.join(here, "vtu_frames.py"), d, out, "--t0", repr(ts),
                                "--keep", ",".join(map(str, keep))] + extra + rest)
            if r.returncode == 0:
                made.append(out)
    final = [m[:-len(".part.npz")] for m in made]
    for old in glob.glob(f"{outbase}_seg*.npz"):     # the old set (also another job's), then the new one in place
        if old not in made:
            os.remove(old)
    for m, f in zip(made, final):
        os.replace(m, f)
    print("packed: " + " ".join(final))


if __name__ == "__main__":
    main()
