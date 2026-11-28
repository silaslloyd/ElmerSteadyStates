#!/usr/bin/env python3
"""
Remesh one converged sweep point with a much finer band at the head of its temperate patch (the cold / temperate boundary on the bed),
transfer the state onto the new mesh and solve it again at the same d (Newton with both masks frozen, the grounding line by contact):
the same steady state at a resolution of FINE m just upstream of the head.

  python3 jobs/gl_remesh_point.py CONFIG STATE [--fine 20] [--up-km 0.3] [--down-km 0.1] [--grade-km 1] [--tag R0025]

CONFIG: the config of the segment the point is from (work/.../refined/<tag>.toml: its mesh, its res_points as the base); STATE: the
point (sweep_*/seg_*/points/state_NNNNN.npz). The head is at x = x_gl - d. The band is FINE m from `up` km upstream of the head
(x < head) to `down` km downstream of it, graded to the config's own resolution over `grade` km on both sides. New mesh, config and
start state: <work>/<tag>/, <work>/refined/<tag>.toml; the solved point: <work>/remesh_<tag>/seg_<tag>/points/state_NNNNN.npz.
Run it on a compute node (jobs/gl_remesh_point.cmd): setup, init, transfer and the solve all start Elmer.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from ess.config import load_config  # noqa: E402
from ess.spectrum import gl_position  # noqa: E402
from ess.statefile import load_state  # noqa: E402
from sweep_refined import banded_points, write_config  # noqa: E402


def run(cmd):
    print("remesh: $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("state")
    ap.add_argument("--fine", type=float, default=20.0, help="horizontal resolution of the band (m)")
    ap.add_argument("--up-km", type=float, default=0.3, help="the band's extent upstream of the head (km)")
    ap.add_argument("--down-km", type=float, default=0.1, help="the band's extent downstream of the head (km)")
    ap.add_argument("--grade-km", type=float, default=1.0, help="grading length to the config's own resolution (km)")
    ap.add_argument("--tag", help="new mesh / config name (default R + the head's position in km, four digits)")
    ap.add_argument("--gl-rounds", type=int, default=6)
    ap.add_argument("--newton-its-max", type=int, default=8)
    ap.add_argument("--shelf-break", default="--strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 --width-km 10")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work = cfg["_work"]
    base = cfg["case"]["res_points"]
    front_km = float(cfg["case"].get("front_km", base[-1][0]))
    s = load_state(a.state)
    d_m = float(s.meta["sweep_d"])
    xgl = gl_position(s.frozen, s.diag) / 1e3
    xh = xgl - d_m / 1e3
    tag = a.tag or f"R{int(round(xh)):04d}"
    pts = banded_points(base, xh - a.up_km, xh + a.down_km, a.fine, a.grade_km, front_km)
    print(f"remesh: d = {d_m / 1e3:g} km, grounding line {xgl:.3f} km, head at x = {xh:.3f} km; band {a.fine:g} m over "
          f"[{xh - a.up_km:.3f}, {xh + a.down_km:.3f}] km, graded over {a.grade_km:g} km; res_points {pts}", flush=True)
    mdir = os.path.join(work, tag)
    cfgf = os.path.join(work, "refined", f"{tag}.toml")
    os.makedirs(os.path.dirname(cfgf), exist_ok=True)
    write_config(a.config, cfgf, pts, "{work}/" + tag)
    print(f"remesh: config {cfgf}", flush=True)
    if a.dry_run:
        return
    py = sys.executable
    st0, st1, st2 = (os.path.join(mdir, f) for f in ("state0.npz", "start0.npz", "start.npz"))
    if not os.path.exists(st0):
        run([py, "-u", "-m", "ess", "setup", cfgf])
        run([py, "-u", "-m", "ess", "init", cfgf, "--out", st0])
    if not os.path.exists(st1):
        run([py, "-u", "-m", "ess", "transfer", cfgf, "--state", st0, "--from", a.state, "--out", st1])
    if not os.path.exists(st2):
        run([py, "jobs/shelf_break.py", st1, st2] + a.shelf_break.split() + ["--force"])
    out = os.path.join(work, f"remesh_{tag}", f"seg_{tag}")
    run([py, "-u", "-m", "ess", "sweep", cfgf, "--state", st2, "--d-start-km", f"{d_m / 1e3:g}", "--d-end-km",
         f"{d_m / 1e3:g}", "--gl-rounds", str(a.gl_rounds), "--newton-its-max", str(a.newton_its_max), "--out", out])
    pt = os.path.join(out, "points", f"state_{int(round(d_m / 100)):05d}.npz")
    print(f"remesh: done: {pt}\n  stability: CFG={cfgf} STATE={pt} M=20 sbatch jobs/gl_stability.cmd", flush=True)


if __name__ == "__main__":
    main()
