#!/usr/bin/env python3
"""A state file from a run's checkpoint (CHECKPOINT_EVERY=N / ESS_CHECKPOINT_EVERY: checkpoint.bin.* in the run
directory, written every N steps, overwritten each time), to continue a long single run that stopped
(START_STATE=OUT.npz), or to look at it with the usual tools (slim_states.py, thermal_start.py, seam_check.py).

    python3 periodic_test/checkpoint_state.py RUNDIR START.npz OUT.npz

START.npz: the state the run started from (the layout of the state vector and the frozen fields not in the
checkpoint). The time of the checkpoint (years since the run's start) from the run's log and sif: the last
multiple of N steps that was finished."""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import find_name, load_state, merge_partitions, partition_files, save_state  # noqa: E402

if len(sys.argv) != 4:
    raise SystemExit(__doc__)
rundir, start, out = sys.argv[1:]
files = partition_files(os.path.join(rundir, "checkpoint.bin"))
if not files:
    raise SystemExit(f"{rundir}: no checkpoint.bin.* (a run with CHECKPOINT_EVERY?)")
rec, mism = merge_partitions(files)
bad = {k: v for k, v in mism.items() if v > 1e-8}
if bad:
    print("shared-node values differ between partitions: " + ", ".join(f"{k} {v:.1e}" for k, v in bad.items()))
s0 = load_state(start)
x = s0.layout.to_vector(rec)
names = {find_name(rec, b.name) for b in s0.layout.blocks}
diag = {k: v for k, v in rec.items() if k not in names}
frozen = dict(s0.frozen)
for nm in ("ColdTempMask", "GroundedMask", "Water Sheet Thickness"):     # the masks the run moved: as it had them
    try:
        kr = find_name(rec, nm)
    except KeyError:
        continue
    for k in [k for k in frozen if k.lower() == nm.lower()]:
        del frozen[k]
    frozen[nm] = rec[kr]
sif = open(os.path.join(rundir, "run.sif")).read()
m = re.search(r'Exec Intervals\(\d+\)\s*=\s*Integer\s+(\d+)[^\n]*\n(\s*Exec Intervals Offset[^\n]*\n)?'
              r'\s*Equation = "StateWriteCheckpoint"', sif)
every = int(m.group(1)) if m else 1
offset = bool(m and m.group(2))           # with the offset: at steps N, 2N, ...; without (older runs): 1, N + 1, ...
m = re.search(r"Timestep Sizes\s*=\s*(?:Real\s*)?([-+0-9.eE]+)", sif)
dt = float(m.group(1)) if m else 1.0
steps = [int(v) for v in re.findall(r"MAIN: Time:\s*(\d+)/", open(os.path.join(rundir, "elmer.log"), errors="replace").read())]
done = (steps[-1] if steps else 0)                          # the last step started; finished if the run ended
ended = "ALL DONE" in open(os.path.join(rundir, "elmer.log"), errors="replace").read()[-5000:]
last = done if ended else done - 1        # the last finished step
k = last // every * every if offset else ((last - 1) // every) * every + 1
t = k * dt
save_state(out, s0.layout, x, s0.p, frozen, diag, time=t)
print(f"{rundir}: checkpoint after step {k} (every {every} x {dt:g} yr: t = {t:g} yr after the run's start; "
      f"last step started {done}) -> {out}")
