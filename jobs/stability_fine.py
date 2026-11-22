#!/usr/bin/env python3
"""Linear stability of a converged sweep point on a FINE mesh over the whole subtemperate region and the grounding line
(are the temperature stripes that grow in the Newton iterates near the onset of subtemperate sliding a physical
instability, Mantelli & Schoof 2019 Part 2, or numerical? Physical modes keep their wavelength when the mesh is refined):

    1. the mesh: --fine m (default 500) from --fine-from-km (default 150: upstream of the onset of sliding) to 1065 km
       (the grounding line band), graded over 10 km, the base config's resolution elsewhere; config
       <work>/refined/L<head>.toml, mesh <work>/L<head>/;
    2. the source state carried onto it (ess transfer), the shelf break as every sweep segment's;
    3. Newton at the source's d with the contact grounding-line rounds (ess sweep --d-start-km d --d-end-km d)
                                                                     -> <work>/stability_fine/L<head>/sweep/
    4. ess stability on the converged state (the physical map [stability_map], Arnoldi with --m products; the patch
       and the grounding line held)                                  -> <work>/stability/L<head>_d<d>/
Every step resumes (its output kept; the Arnoldi checkpoint).

    python3 jobs/stability_fine.py CONFIG [--from-state S.npz --d-km D] [--fine 500] [--fine-from-km 150] [--m 30]

Default source: the converged point with the smallest d in <work>/sweep_refined/seg_*.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sweep_refined import converged, last_converged, write_config  # noqa: E402
from ess.config import load_config  # noqa: E402
from ess.spectrum import gl_position  # noqa: E402
from ess.statefile import load_state  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} stability-fine: {msg}", flush=True)


def run(cmd):
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--from-state")
    ap.add_argument("--d-km", type=float)
    ap.add_argument("--fine", type=float, default=500.0)
    ap.add_argument("--fine-from-km", type=float, default=150.0)
    ap.add_argument("--gl-fine", type=float, default=250.0)
    ap.add_argument("--ct-fine", type=float, default=250.0, help="m at the cold/temperate transition (the head)")
    ap.add_argument("--ct-pad-km", type=float, default=10.0, help="--ct-fine over head +- this (graded over 10 km)")
    ap.add_argument("--m", type=int, default=30)
    ap.add_argument("--map-steps", type=int, help="Newton's map for the re-convergence: this many [map] dt steps (the "
                    "stability analysis uses [stability_map] either way)")
    ap.add_argument("--tag-suffix", default="", help="appended to the mesh/output tag (a copy on another rank count)")
    ap.add_argument("--gl-rounds", type=int, default=6)
    ap.add_argument("--shelf-break", default="--strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 --width-km 10")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work = cfg["_work"]
    if a.from_state:
        if a.d_km is None:
            raise SystemExit("--from-state needs --d-km")
        src, d = a.from_state, a.d_km
    else:
        pts = [p for f in sorted(glob.glob(os.path.join(work, "sweep_refined", "seg_*"))) for p in converged(f)]
        if not pts:
            raise SystemExit("no converged point in sweep_refined")
        d, src = min(pts)
    s0 = load_state(src)
    xgl = gl_position(s0.frozen, s0.diag) / 1e3
    head = xgl - d
    tag = f"L{int(round(head)):04d}{a.tag_suffix}"     # (a suffix: a second copy, e.g. on another rank count)
    log(f"source {os.path.relpath(src, ROOT)}: d = {d:g} km, head {head:.2f} km, grounding line {xgl:.2f} km")
    x0 = a.fine_from_km
    hp = a.ct_pad_km                    # the cold/temperate transition (the patch head): --ct-fine over head +- hp
    pts = [[0.0, 2000.0], [x0 - 10.0, 2000.0], [x0, a.fine], [head - hp - 10.0, a.fine], [head - hp, a.ct_fine],
           [head + hp, a.ct_fine], [head + hp + 10.0, a.fine], [1005.0, a.fine], [1015.0, a.gl_fine],
           [1065.0, a.gl_fine], [1080.0, 1000.0], [float(cfg["case"]["res_points"][-1][0]), 1000.0]]
    cfgf = os.path.join(work, "refined", f"{tag}.toml")
    os.makedirs(os.path.dirname(cfgf), exist_ok=True)
    write_config(a.config, cfgf, pts, "{work}/" + tag, None, a.map_steps)
    log(f"mesh {tag}: {a.fine:g} m over {x0:g} .. 1005 km, {a.ct_fine:g} m over the head {head - hp:.1f} .. "
        f"{head + hp:.1f} km, {a.gl_fine:g} m over 1015 .. 1065 km; res_points {pts}")
    mdir = os.path.join(work, tag)
    py = sys.executable
    st0, st1, st2 = (os.path.join(mdir, f) for f in ("state0.npz", "start0.npz", "start.npz"))
    if not os.path.exists(st0):
        run([py, "-u", "-m", "ess", "setup", cfgf])
        run([py, "-u", "-m", "ess", "init", cfgf, "--out", st0])
    if not os.path.exists(st1):
        run([py, "-u", "-m", "ess", "transfer", cfgf, "--state", st0, "--from", src, "--out", st1])
    if not os.path.exists(st2):
        run([py, "jobs/shelf_break.py", st1, st2] + a.shelf_break.split() + ["--force"])
    sdir = os.path.join(work, "stability_fine", tag, "sweep")
    got, _ = last_converged(sdir)
    if got is None:
        run([py, "-u", "-m", "ess", "sweep", cfgf, "--state", st2, "--d-start-km", f"{d:g}", "--d-end-km", f"{d:g}",
             "--gl-rounds", str(a.gl_rounds), "--out", sdir])
        got, _ = last_converged(sdir)
        if got is None:
            raise SystemExit("Newton did not converge on the fine mesh")
    # a distinct name: ess stability writes into <work>/stability/<state file name>/
    named = os.path.join(work, "stability_fine", tag, f"{tag}_d{int(round(d * 10)):05d}.npz")
    if not os.path.exists(named):
        shutil.copyfile(got[1], named)
    log(f"converged on the fine mesh: {os.path.relpath(got[1], ROOT)}; the stability analysis (m = {a.m})")
    run([py, "-u", "-m", "ess", "stability", cfgf, "--state", named, "--m", str(a.m)])
    log("done")


if __name__ == "__main__":
    main()
