#!/usr/bin/env python3
"""All the outputs of one job, across its remeshes: every run directory of the job (runs/*_job<JOB>_*, in time
order: one per mesh with `ess run --remesh-auto`) packed with plots/vtu_frames.py into OUTBASE_seg<k>.npz, each with
its true model time (--t0: the times the earlier segments ran, from their logs' last "MAIN: Time" line, plus T0).
plots/plot_state_3d.py takes them all at once (one page, the mesh changing where the run remeshed):

    python3 plots/job_frames.py WORKDIR JOB OUTBASE [--t0 T0] [vtu_frames options: --every N, --every-yr Y, ...]
    python3 plots/plot_state_3d.py OUTBASE_seg*.npz ...

Segments without VTU output (e.g. a run the trigger ended before its first output) are skipped, their time counted.
"""
import glob
import os
import re
import subprocess
import sys


def main():
    a = sys.argv[1:]
    if len(a) < 3:
        raise SystemExit(__doc__)
    work, job, outbase = a[0], a[1], a[2]
    rest = a[3:]
    t0 = 0.0
    if "--t0" in rest:
        k = rest.index("--t0")
        t0 = float(rest[k + 1])
        del rest[k:k + 2]
    dirs = sorted(glob.glob(os.path.join(work, "runs", f"*_job{job}_*")))
    if not dirs:
        raise SystemExit(f"no run directories of job {job} in {work}/runs")
    off = t0
    made = []
    here = os.path.dirname(os.path.abspath(__file__))
    for k, d in enumerate(dirs, 1):
        log = os.path.join(d, "elmer.log")
        tend = 0.0
        if os.path.exists(log):
            for line in open(log, errors="replace"):
                m = re.match(r"\s*MAIN: Time: \d+/\d+:\s+(\S+)", line)
                if m:
                    tend = float(m.group(1))
        nv = len(glob.glob(os.path.join(d, "base_t*.pvtu")))
        print(f"segment {k}: {os.path.basename(d)}: {nv} outputs, t {off:g} .. {off + tend:g} yr")
        if nv:
            out = f"{outbase}_seg{k}.npz"
            r = subprocess.run([sys.executable, os.path.join(here, "vtu_frames.py"), d, out, "--t0", repr(off)] + rest)
            if r.returncode == 0:
                made.append(out)
        off += tend
    print("packed: " + " ".join(made))


if __name__ == "__main__":
    main()
