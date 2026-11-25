#!/usr/bin/env python3
"""
A 3D land run's progress as ONE short text block to paste into a chat (a phone, no file transfer): per time step the
first-iteration change of the Stokes velocity and of the free surfaces and the number of coupled iterations (from the
run's elmer.log), and, if a frames file is given, the surface and base along the first row of nodes at the saved
frames (5 km grid, whole metres). zlib + base64 in one line between BEGIN and END.

    python3 plots/land_compact.py RUNDIR [FRAMES.npz] [--t0 YEARS] [--dx 5] [--every N]

RUNDIR: the run directory (RUNDIR/elmer.log). FRAMES.npz: from plots/vtu_frames.py (its times already include its
--t0). --t0: the run's start time (yr), added to the steps' times (default 0). --every N: keep every N-th step.
"""
import base64
import json
import os
import re
import sys
import zlib

import numpy as np


def steps_of(log, t0, every):
    out, cur = [], None
    st = {}
    with open(log, errors="replace") as f:
        for line in f:
            m = re.match(r"MAIN: Time:\s*(\d+)/(\d+):\s*([0-9.Ee+-]+)", line)
            if m:
                if cur is not None:
                    out.append(cur)
                n, tot, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
                cur = {"n": n, "t": t0 + t, "dt": t / n, "stokes": None, "surf": None, "its": 0}
                continue
            if cur is None or "ComputeChange: NS" not in line:
                continue
            m = re.search(r"\(ITER=(\d+)\)\s*\(NRM,RELC\):\s*\(\s*([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s*\)\s*::\s*(.+?)\s*$", line)
            if not m:
                continue
            it, rel, what = int(m.group(1)), float(m.group(3)), m.group(4)
            if what.startswith("free surface"):
                cur["its"] = max(cur["its"], it)
                if it == 1:
                    cur["surf"] = rel
            elif what.startswith("stokes") and it == 1:
                cur["stokes"] = rel
    if cur is not None:
        out.append(cur)
    out = out[:-1] if out and cur["surf"] is None else out      # the step still running
    return [s for s in out if s["n"] % every == 0 or s is out[-1]]


def frames_of(path, dx):
    d = np.load(path, allow_pickle=True)
    t = [float(v) for v in d["frame_times"]]
    tx, ty, tz = d["top_x"], d["top_y"], d["top_z"]
    bx, by, bz = d["base_x"], d["base_y"], d["base_z"]
    row = np.where(np.abs(ty - ty.min()) < 1e-6 * max(1.0, np.abs(ty).max()) + 1e-3)[0]
    row = row[np.argsort(tx[row])]
    brow = np.where(np.abs(by - by.min()) < 1e-6 * max(1.0, np.abs(by).max()) + 1e-3)[0]
    brow = brow[np.argsort(bx[brow])]
    xg = np.arange(0.0, tx[row].max() / 1e3 + 1e-9, dx)
    zs = [[int(round(v)) for v in np.interp(xg * 1e3, tx[row], tz[k][row])] for k in range(len(t))]
    zb = [[int(round(v)) for v in np.interp(xg * 1e3, bx[brow], bz[k][brow])] for k in range(len(t))]
    return {"dx": dx, "t": t, "zs": zs, "zb": zb}


def main():
    a = sys.argv[1:]
    opt = {"--t0": 0.0, "--dx": 5.0, "--every": 1, "--first-frame": 60}
    for k in list(opt):
        if k in a:
            i = a.index(k)
            opt[k] = type(opt[k])(a[i + 1])
            del a[i:i + 2]
    if not a:
        raise SystemExit(__doc__)
    steps = steps_of(a[0].rstrip("/") + "/elmer.log", opt["--t0"], opt["--every"])
    rec = {"steps": [[s["n"], round(s["t"], 1), s["dt"], s["stokes"], s["surf"], s["its"]] for s in steps]}
    if len(a) > 1 and not os.path.exists(a[1]):
        print(f"no frames file {a[1]} yet (the first frame is written at step {opt['--first-frame']}): steps only", file=sys.stderr)
    elif len(a) > 1:
        rec["frames"] = frames_of(a[1], opt["--dx"])
    blob = base64.b64encode(zlib.compress(json.dumps(rec, separators=(",", ":")).encode(), 9)).decode()
    print(f"{len(steps)} steps" + (f", {len(rec['frames']['t'])} frames" if "frames" in rec else "") + f", {len(blob)} characters")
    print("BEGIN")
    print(blob)
    print("END")


if __name__ == "__main__":
    main()
