#!/usr/bin/env python3
"""
The VTU output of one long Elmer run (ESS_VTU_EVERY / jobs VTU_EVERY=N: base_t*.pvtu and surface_t*.pvtu every N
steps, one piece per rank) packed into ONE small file for plots/plot_state_3d.py: the surface and the base of every
output step, nodes shared between partitions once, in the same (x, y) order in every frame, float32.

    python3 plots/vtu_frames.py RUNDIR [OUT.npz] [--t0 YEARS] [--tmax YEARS]
        (default OUT: RUNDIR/../<run dir name>_frames.npz; --t0: added to the times, e.g. a run continued from a
        checkpoint at year 1000; --tmax: only the frames up to that time (after --t0), e.g. to join two runs)
    python3 plots/plot_state_3d.py OUT.npz [--full ...]    (on the Mac: one frame per output step on the slider)

Re-running it adds the steps written since (it rereads everything: a few seconds per step). The time of frame k is
k * N * dt, N and dt from the run's run.sif (Exec Intervals of the "vtu base" solver, Timestep Sizes).
Fields: x, y, z (the surface / base elevation), velocity (vx, vy, vz), temperature; on the base also the grounded
mask and the bedrock, and when the run wrote them ColdTempMask, Water Sheet Thickness, Melt Rate (THERMAL=nowater)
and CFL, Step Change (STEP_CHECK=1).
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_state_3d import read_vtu, vtu_steps  # noqa: E402

EXTRA = {"coldtempmask": "ct", "water sheet thickness": "hw", "melt rate": "melt", "cfl": "cfl", "step change": "dz",
         "groundedmask": "mask", "bedrock": "bed", "temperature": "T"}


def run_timing(rundir):
    """(output interval in steps, dt in years) from the run's sif"""
    sif = open(os.path.join(rundir, "run.sif")).read()
    m = re.search(r'Exec Intervals\(\d+\)\s*=\s*Integer\s+(\d+)[^\n]*\n(\s*Exec Intervals Offset[^\n]*\n)?'
                  r'\s*Equation = "vtu base"', sif)
    every = int(m.group(1)) if m else 1
    offset = bool(m and m.group(2))       # with the offset: output k at step k N; without (older runs): (k-1) N + 1
    m = re.search(r"Timestep Sizes\s*=\s*(?:Real\s*)?([-+0-9.eE]+)", sif)
    dt = float(m.group(1)) if m else 1.0
    return every, dt, offset


def surface(path):
    """one surface (a .pvtu): nodes once, sorted by (x, y); {x, y, z, vx, vy, vz, extras}"""
    X, D = read_vtu(path)
    if not len(X):
        return None
    key = np.c_[np.round(X[:, 0], 0), np.round(X[:, 1], 0)]
    _, first = np.unique(key, axis=0, return_index=True)       # sorted by (x, y), shared nodes once
    out = dict(x=X[first, 0], y=X[first, 1], z=X[first, 2])
    u = D.get("velocity")
    if u is not None:
        for c, n in enumerate(("vx", "vy", "vz")):
            out[n] = u[first, c]
    for name, short in EXTRA.items():
        if name in D:
            out[short] = D[name][first, 0]
    return out


def main():
    args = sys.argv[1:]
    t0, tmax = 0.0, None
    for opt in ("--t0", "--tmax"):
        if opt in args:
            k = args.index(opt)
            v = float(args[k + 1])
            args = args[:k] + args[k + 2:]
            if opt == "--t0":
                t0 = v
            else:
                tmax = v
    if not args:
        raise SystemExit(__doc__)
    sys.argv = [sys.argv[0]] + args
    rundir = sys.argv[1].rstrip("/")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(rundir),
                                                             os.path.basename(rundir) + "_frames.npz")
    steps = vtu_steps(rundir)
    if not steps:
        raise SystemExit(f"{rundir}: no base_t*.pvtu (a run with VTU_EVERY?)")
    every, dt, offset = run_timing(rundir)
    step_of = (lambda k: k * every) if offset else (lambda k: (k - 1) * every + 1)
    frames = {"top": [], "base": []}
    times = []
    for f in steps:
        try:
            b = surface(f)
            t = surface(f.replace("base_t", "surface_t"))
        except Exception as e:                    # the step being written right now
            print(f"  {os.path.basename(f)}: not readable ({e}), stopping there")
            break
        if b is None or t is None:
            break
        if frames["base"] and (len(b["x"]) != len(frames["base"][0]["x"]) or len(t["x"]) != len(frames["top"][0]["x"])):
            print(f"  {os.path.basename(f)}: a different number of nodes, stopping there")
            break
        frames["base"].append(b)
        frames["top"].append(t)
        k = int(re.findall(r"_t(\d+)", os.path.basename(f))[0])
        if tmax is not None and t0 + step_of(k) * dt > tmax + 1e-9:
            frames["base"].pop(); frames["top"].pop()
            break
        times.append(t0 + step_of(k) * dt)
    if not times:
        raise SystemExit("no readable steps")
    rec = {"frame_times": np.array(times), "frame_every": every, "frame_dt": dt, "frame_run": rundir}
    for side in ("top", "base"):
        fr = frames[side]
        rec[f"{side}_x"], rec[f"{side}_y"] = fr[0]["x"], fr[0]["y"]
        for n in sorted(set.intersection(*[set(d) for d in fr]) - {"x", "y"}):
            if n == "bed":
                rec[f"{side}_bed"] = fr[0]["bed"].astype(np.float32)
            else:
                rec[f"{side}_{n}"] = np.stack([d[n] for d in fr]).astype(np.float32)
    np.savez_compressed(out, **rec)
    print(f"{rundir}: {len(times)} steps (every {every} x {dt:g} yr, t {times[0]:g} .. {times[-1]:g} yr), "
          f"{len(rec['top_x'])} surface / {len(rec['base_x'])} base nodes -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
