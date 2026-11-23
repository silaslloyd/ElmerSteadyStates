#!/usr/bin/env python3
"""LATERAL (spanwise) stability of a converged flowline steady state: the state extruded across a wide 3D strip
(periodic sides, y-uniform), re-converged there, then `ess stability --spanwise`: Arnoldi on the across-flow
modes only (the across-flow mean, n = 0 = the flowline modes, projected out); each mode
delta_hat(x, z) cos/sin(2 pi n y / W), its dominant n and wavelength W / n reported (ess/spanwise.py). --walls:
free-slip side walls instead (cos(n pi y / W), wavelength 2 W / n).

    1. config <work>/refined/Z<head>.toml: the base config with [case] width = --width-km, y_elements = --ny, the
       mesh STRUCTURED across the flow (mesh_across = "structured": --ny rows of width/ny, every node line across at
       one x, as ess/spanwise.py needs), the calving front --shelf-km (15) beyond the grounding line (the steady
       unbuttressed shelf beyond carries no information the front's sea pressure does not), periodic sides (METIS with periodic partners together), res_points --fine-x m from --fine-from-km to 1005 km, --head-fine m over the head
       +- 10 km, 250 m over the grounding line band; Newton's map --map-steps; with --solver iterative (default) the
       Stokes P1 + bubbles (ess init: a cheap inexact solve; Newton and stability: GCR + --stokes-prec ILU0 to
       --stokes-tol 1e-8), heat Krylov + ILU to 1e-10 (a direct MUMPS solve grows much faster than the node count in 3D);
    2. ess setup / init, the state carried across (by x columns: y-uniform), the shelf break;
    3. Newton at the source's d (the contact grounding-line rounds): a y-uniform steady state stays one, only the
       remeshing to correct                                         -> <work>/stability_spanwise/Z<head>/sweep/
    4. ess stability --spanwise (the physical map [stability_map], Arnoldi --m)
                                                                    -> <work>/stability/Z<head>_d<d>_spanwise/
--stop-after mesh|newton: stop there (e.g. to time one map run on the wide strip first). Resumes.

    python3 jobs/stability_spanwise.py CONFIG --from-state S.npz --d-km D [--width-km 100] [--ny 50] [--walls]
            [--fine-x 1000] [--fine-from-km 150] [--head-fine 250] [--map-steps 10] [--m 30]
            [--solver iterative|direct] [--stop-after mesh|newton]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sweep_refined import last_converged, write_config  # noqa: E402
from ess.config import load_config  # noqa: E402
from ess.spectrum import gl_position  # noqa: E402
from ess.statefile import load_state  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} stability-spanwise: {msg}", flush=True)


def run(cmd, env=None):
    log("$ " + " ".join(cmd) + (f"   [{' '.join(f'{k}={v}' for k, v in env.items())}]" if env else ""))
    if subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, **env) if env else None).returncode != 0:
        raise SystemExit("step failed")


def set_key(text, section, key, value):
    """key = value in [section] (replaced, or added at the section's start)"""
    m = re.search(rf"^\[{re.escape(section)}\]\s*$(.*?)(?=^\[|\Z)", text, flags=re.M | re.S)
    if not m:
        return text + f"\n[{section}]\n{key} = {value}\n"
    body, n = re.subn(rf"^{re.escape(key)}\s*=.*$", f"{key} = {value}", m.group(1), count=1, flags=re.M)
    if n == 0:
        body = f"\n{key} = {value}" + body
    return text[:m.start(1)] + body + text[m.end(1):]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--from-state", required=True)
    ap.add_argument("--d-km", type=float, required=True)
    ap.add_argument("--width-km", type=float, default=100.0)
    ap.add_argument("--ny", type=int, default=50)
    ap.add_argument("--walls", action="store_true", help="free-slip side walls instead of periodic sides")
    ap.add_argument("--fine-x", type=float, default=1000.0)
    ap.add_argument("--fine-from-km", type=float, default=150.0)
    ap.add_argument("--head-fine", type=float, default=250.0)
    ap.add_argument("--map-steps", type=int, default=10)
    ap.add_argument("--m", type=int, default=30)
    ap.add_argument("--solver", choices=("iterative", "direct"), default="iterative")
    ap.add_argument("--gl-rounds", type=int, default=6)
    ap.add_argument("--shelf-km", type=float, default=15.0,
                    help="the calving front this far beyond the source's grounding line (0: the config's front)")
    ap.add_argument("--stokes-prec", default="ILU0", help="iterative Stokes preconditioner of the Newton/stability runs")
    ap.add_argument("--stokes-tol", type=float, default=1e-8, help="their Stokes tolerance (heat stays at 1e-10)")
    ap.add_argument("--stop-after", choices=("mesh", "newton"))
    ap.add_argument("--shelf-break", default="--strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 --width-km 10")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work = cfg["_work"]
    s0 = load_state(a.from_state)
    xgl = gl_position(s0.frozen, s0.diag) / 1e3
    d = a.d_km
    head = xgl - d
    # short (Elmer reads at most 128 characters of the mesh path): Z<head>, then only what differs from the defaults
    tag = (f"Z{int(round(head)):04d}" + ("" if a.width_km == 100.0 else f"w{a.width_km:g}")
           + ("" if a.ny == 50 else f"n{a.ny}") + ("W" if a.walls else "")
           + ("" if a.shelf_km == 15.0 else f"s{a.shelf_km:g}")
           + ("" if a.fine_x == 1000.0 else f"x{a.fine_x:g}")).replace(".", "p")
    x0, f, hf = a.fine_from_km, a.fine_x, a.head_fine
    # the shelf cut: the calving front --shelf-km beyond the grounding line (sea pressure there, as at the full
    # domain's front). A steady unbuttressed shelf (laterally uniform, no friction, periodic sides or free-slip walls)
    # passes the same depth-integrated force through every cross-section, so the front's sea pressure at the cut gives
    # the force the rest of the shelf would; only a boundary layer of a few ice thicknesses next to the cut differs.
    # The full floating shelf in full Stokes made GCR + ILU stall (relative residual 1.0: with periodic sides nothing
    # holds the floating ice across the flow); the SSA-shelf runs had only a 10 km floating strip in Stokes.
    front = float(cfg["case"].get("front_km", cfg["case"]["res_points"][-1][0]))
    if a.shelf_km > 0:
        front = min(front, round(xgl + a.shelf_km, 1))
    if front <= 1015.0 + 1.0:
        raise SystemExit(f"the front at {front:g} km leaves no room for the 250 m grounding-line band from 1015 km")
    pts = [[0.0, 2000.0], [x0 - 10.0, 2000.0], [x0, f], [head - 20.0, f], [head - 10.0, hf], [head + 10.0, hf],
           [head + 20.0, f], [1005.0, f], [1015.0, 250.0]]
    pts += ([[1065.0, 250.0], [1080.0, 1000.0], [front, 1000.0]] if front > 1085.0 else [[front, 250.0]])
    cfgf = os.path.join(work, "refined", f"{tag}.toml")
    os.makedirs(os.path.dirname(cfgf), exist_ok=True)
    write_config(a.config, cfgf, pts, "{work}/" + tag, None, a.map_steps)
    text = open(cfgf).read()
    text = set_key(text, "case", "width", f"{a.width_km * 1e3:.1f}")
    text = set_key(text, "case", "y_elements", str(a.ny))
    text = set_key(text, "case", "periodic_sides", "false" if a.walls else "true")
    text = set_key(text, "case", "mesh_across", '"structured"')
    text = set_key(text, "case", "front_km", f"{front:g}")
    if not a.walls:
        text = set_key(text, "case", "partitioner", '"metisper"')
    if a.solver == "iterative":
        # P1 + bubbles. `ess init` (whose velocity is not used): the cheap inexact Stokes solve (elmer/stokes_cheap.sif.in,
        # GCR + Jacobi, 20 iterations), as the 3D strip configs: on the 100 km x 50 row strip at 256 ranks P2/P1 + ILU4,
        # bubbles + ILU8 and + ILU4 overflowed Elmer's ILU, and GCR + ILU1 made no progress at all from the initial
        # conditions (relative residual 1.0 through 5000 iterations). The Newton and stability runs: GCR + --stokes-prec
        # (ILU0, as the marine strip runs of jobs/scaling_sweep.sh) to --stokes-tol, set through ESS_LINEAR_STOKES /
        # ESS_STOKES_PREC for those steps only.
        for k, v in (("linear_stokes", '"block"'), ("stokes_block_file", '"elmer/stokes_cheap.sif.in"'),
                     ("linear_heat", '"iterative"'), ("iterative_tol", "1e-10"), ("stokes_element", '"Bubbles"'),
                     ("stokes_iterative_tol", f"{a.stokes_tol:g}")):
            text = set_key(text, "elmer", k, v)
    open(cfgf, "w").write(text)
    lam = [a.width_km * (2.0 if a.walls else 1.0) / n for n in (1, 2, 3, 4)]
    log(f"shelf cut: the calving front at {front:g} km, {front - xgl:.1f} km beyond the grounding line ({xgl:.2f} km)")
    log(f"source {a.from_state}: d {d:g} km, head {head:.2f} km; strip {a.width_km:g} km x {a.ny} rows "
        f"({a.width_km * 1e3 / a.ny:g} m across, {'free-slip walls' if a.walls else 'periodic sides'}: wavelengths "
        f"{', '.join(f'{v:.4g}' for v in lam)} ... km), {f:g} m along over {x0:g} .. 1005 km, {hf:g} m at the head, "
        f"{a.solver} solves; config {cfgf}")
    renv = ({"ESS_LINEAR_STOKES": "iterative", "ESS_STOKES_PREC": a.stokes_prec}
            if a.solver == "iterative" else None)
    mdir = os.path.join(work, tag)
    py = sys.executable
    st0, st1, st2 = (os.path.join(mdir, n) for n in ("state0.npz", "start0.npz", "start.npz"))
    if not os.path.exists(st0):
        run([py, "-u", "-m", "ess", "setup", cfgf])
        run([py, "-u", "-m", "ess", "init", cfgf, "--out", st0])
    if not os.path.exists(st1):
        run([py, "-u", "-m", "ess", "transfer", cfgf, "--state", st0, "--from", a.from_state, "--out", st1])
    if not os.path.exists(st2):
        run([py, "jobs/shelf_break.py", st1, st2] + a.shelf_break.split() + ["--force"])
    if a.stop_after == "mesh":
        log("stopping after the mesh (--stop-after mesh)")
        return
    sdir = os.path.join(work, "stability_spanwise", tag, "sweep")
    got, _ = last_converged(sdir)
    if got is None:
        run([py, "-u", "-m", "ess", "sweep", cfgf, "--state", st2, "--d-start-km", f"{d:g}", "--d-end-km", f"{d:g}",
             "--gl-rounds", str(a.gl_rounds), "--out", sdir], renv)
        got, _ = last_converged(sdir)
        if got is None:
            raise SystemExit("Newton did not converge on the wide strip")
    if a.stop_after == "newton":
        log(f"stopping after Newton (--stop-after newton): {got[1]}")
        return
    named = os.path.join(work, "stability_spanwise", tag, f"{tag}_d{int(round(d * 10)):05d}.npz")
    if not os.path.exists(named):
        shutil.copyfile(got[1], named)
    run([py, "-u", "-m", "ess", "stability", cfgf, "--state", named, "--m", str(a.m), "--spanwise"], renv)
    log("done")


if __name__ == "__main__":
    main()
