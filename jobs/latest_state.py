#!/usr/bin/env python3
"""The state of a job's newest run directory at its last checkpoint (periodic_test/checkpoint_state.py), without
having to know which state that run started from: the start state is the last "run: starting on / continuing on M
from S" line of the job's logs (a remesh), else the job's START_STATE ("runs continue from S" in its .out). Prints
the mesh and the model time.

    python3 jobs/latest_state.py WORKDIR [JOB] [OUT.npz]     (JOB: default the newest run directory's job;
                                                              OUT: default <work>/latest_<job>.npz)
"""
import glob
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)


def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    work = a[0].rstrip("/")
    runs = sorted(glob.glob(os.path.join(work, "runs", "*_job*")), key=os.path.getmtime)
    job = a[1] if len(a) > 1 else re.search(r"_job(\d+)_", runs[-1]).group(1)
    dirs = sorted(glob.glob(os.path.join(work, "runs", f"*_job{job}_*")))
    if not dirs:
        raise SystemExit(f"no run directories of job {job}")
    rundir = dirs[-1]
    out = a[2] if len(a) > 2 else os.path.join(work, f"latest_{job}.npz")
    start, mesh = None, None
    for f in sorted(glob.glob(os.path.join(root, "slurms", f"scaling_{job}_*.log"))):
        for line in open(f, errors="replace"):
            m = re.search(r"run: (?:starting|continuing) on (\S+) from (\S+?)(?: \(|,|$)", line)
            if m:
                mesh, start = m.group(1), m.group(2)
    if start is None:
        o = os.path.join(root, "slurms", f"scaling_{job}.out")
        for line in open(o, errors="replace"):
            m = re.search(r"runs continue from (\S+?):", line)
            if m:
                start = m.group(1)
        if start is None:
            raise SystemExit(f"job {job}: no start state found in its logs")
    if not os.path.isabs(start) and not os.path.exists(start):
        start = os.path.join(root, start)
    r = subprocess.run([sys.executable, os.path.join(root, "periodic_test", "checkpoint_state.py"), rundir, start, out])
    if r.returncode:
        raise SystemExit(r.returncode)
    sys.path.insert(0, root)
    from ess.statefile import load_state
    s = load_state(out)
    sif = open(os.path.join(rundir, "run.sif")).read()     # the mesh this run is on: its sif's Mesh DB
    m = re.search(r'Mesh DB\s+"[^"]*"\s+"([^"]+)"', sif)
    mesh = m.group(1) if m else (mesh or "?")
    print(f"job {job}: {rundir}, started from {start}; mesh {mesh}; t = {float(s.meta.get('time', 0)):g} yr -> {out}")
    print(f"MESH={mesh}")


if __name__ == "__main__":
    main()
