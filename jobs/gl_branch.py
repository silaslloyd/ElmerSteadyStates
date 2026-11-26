#!/usr/bin/env python3
"""A BRANCH from the converged S1 state of the flowline pipeline with a different grounding-line mesh resolution: does
the steady state stay converged when the grounding line is resolved more finely?

The S1b recipe of jobs/flowline_pipeline.py (shelf cut --cut-shelf-km past the grounding line, a band of --res metres
centred on it, the state carried across with ess transfer and jobs/shelf_break.py), then the S1 transient (Stokes + free
surface, temperate bed, contact grounding line every step) in batches, logging per batch: rms change, grounding
line, how far it moved in the batch and since the start of the branch, and the mass balance |q / (a x) - 1|.
Finishes when rms < --rms-tol, the grounding line moved <= --gl-tol-km and the mass balance <= --flux-tol in
--stable-batches batches in a row, or after --max-chunks chunks (1 chunk = --steps x --dt = 1000 yr).
Run the same with --res 250 (the pipeline's own band: the control: the cut and remesh alone) and --res 125, 500, ...:
a grounding line that differs between the 250 m control and the finer branch has not converged with the mesh.

    python3 jobs/gl_branch.py --config configs/flowline_gl_fine.toml --res 125
    sbatch jobs/gl_branch.cmd (RES=125 CONFIG=...)

Files: work/<case>/branch_<tag>/ (start, transient.npz and its chunk states), config work/<case>/refined/branch_<tag>.toml,
flag file chunks.txt (resumable: run it again). tag = r<res> (--tag to change)."""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from ess.config import load_config  # noqa: E402
from ess.spectrum import gl_position  # noqa: E402
from ess.statefile import load_state  # noqa: E402
from flowline_pipeline import PY, log, mass_balance_dev, run  # noqa: E402
from sweep_refined import banded_points, write_config  # noqa: E402


def gl_km(path):
    s = load_state(path)
    return gl_position(s.frozen, s.diag) / 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/flowline_gl_fine.toml", help="the pipeline's config (its --config)")
    ap.add_argument("--from", dest="src", default=None,
                    help="the converged S1 state (default <work>/pipeline/transient.npz; needs flags/s1_done unless --force)")
    ap.add_argument("--res", type=float, default=125.0, help="band resolution in m")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cut-shelf-km", type=float, default=200.0)
    ap.add_argument("--band-below-km", type=float, default=70.0)
    ap.add_argument("--band-above-km", type=float, default=50.0)
    ap.add_argument("--grade-km", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--max-chunks", type=int, default=40)
    ap.add_argument("--rms-tol", type=float, default=1e-3)
    ap.add_argument("--gl-tol-km", type=float, default=0.1)
    ap.add_argument("--flux-tol", type=float, default=0.01)
    ap.add_argument("--stable-batches", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="branch from the state even if S1 has not finished")
    a = ap.parse_args()
    os.environ.setdefault("ESS_CONTACT_FUNCTION", "elmerstreams")
    cfg = load_config(a.config)
    work = cfg["_work"]
    tag = a.tag or f"r{a.res:g}"
    src = a.src or os.path.join(work, "pipeline", "transient.npz")
    if not a.force and not os.path.exists(os.path.join(work, "pipeline", "flags", "s1_done")):
        sys.exit(f"S1 has not finished ({work}/pipeline/flags/s1_done missing): the state is not converged; --force to branch anyway")
    bdir = os.path.join(work, f"branch_{tag}")
    os.makedirs(bdir, exist_ok=True)
    cfgb = os.path.join(work, "refined", f"branch_{tag}.toml")
    f = {k: os.path.join(bdir, v) for k, v in dict(init="state_init.npz", raw="start_raw.npz", start="start.npz",
                                                  trans="transient.npz", chunks="chunks.txt").items()}
    s = load_state(src)
    xgl = gl_position(s.frozen, s.diag) / 1e3
    front = float(int(round(xgl + a.cut_shelf_km)))
    if not os.path.exists(f["start"]):
        pts = banded_points([[0.0, 2000.0], [front, 2000.0]], xgl - a.band_below_km, xgl + a.band_above_km, a.res,
                            a.grade_km, front)
        log(f"BRANCH {tag}: grounding line {xgl:.2f} km, front {front:g} km, band {a.res:g} m, res_points {pts}")
        os.makedirs(os.path.dirname(cfgb), exist_ok=True)
        write_config(a.config, cfgb, pts, f"{{work}}/branchcase_{tag}")   # mesh dir apart from the states
        text = open(cfgb).read()
        text, n = re.subn(r"^front_km\s*=.*$", f"front_km = {front:g}", text, count=1, flags=re.M)
        if n != 1:
            sys.exit("no front_km line in the config")
        open(cfgb, "w").write(text)
        run([PY, "-u", "-m", "ess", "setup", cfgb])
        run([PY, "-u", "-m", "ess", "init", cfgb, "--out", f["init"]])
        run([PY, "-u", "-m", "ess", "transfer", cfgb, "--state", f["init"], "--from", src, "--out", f["raw"]])
        run([PY, "-u", "jobs/shelf_break.py", f["raw"], f["start"], "--strip-divide-r-km", "1520.8",
             "--r-break-km", "450", "--factor", "30", "--width-km", "10", "--force"])
    gl0 = gl_km(f["start"])
    log(f"BRANCH {tag}: start state on the {a.res:g} m mesh: grounding line {gl0:.2f} km (source {xgl:.2f} km)")
    chunks = int(open(f["chunks"]).read()) if os.path.exists(f["chunks"]) else 0
    if a.max_chunks <= 0:                                       # only the remeshed / cut start state (jobs/gl_cut_shelf.cmd)
        log(f"BRANCH {tag}: --max-chunks {a.max_chunks}: the start state only, no transient: {f['start']}")
        return
    stable = 0
    while chunks < a.max_chunks:
        cur = f["trans"] if os.path.exists(f["trans"]) else f["start"]
        before = gl_km(cur)
        rms = run([PY, "-u", "-m", "ess", "run", cfgb, "--state", cur, "--stokes-only", "--gl", "contact",
                   "--dt", f"{a.dt:g}", "--steps", str(a.steps), "--chunks", str(a.batch), "--save-every", "5",
                   "--chunk-offset", str(chunks), "--linear-stokes", "direct", "--out", f["trans"]],
                  env={"ESS_SLIDING": "temperate"}, capture_rms=True)
        chunks += a.batch
        open(f["chunks"], "w").write(str(chunks))
        now = gl_km(f["trans"])
        try:
            dev = mass_balance_dev(f["trans"])
        except Exception as ex:                                  # noqa: BLE001
            dev = None
            log(f"BRANCH {tag}: mass balance not computed ({ex!r})")
        log(f"BRANCH {tag}: {chunks} chunks ({chunks * a.steps * a.dt:g} yr): rms change {rms}, grounding line {now:.2f} km "
            f"({now - before:+.2f} in this batch, {now - gl0:+.2f} since the branch start), mass balance "
            f"{'n/a' if dev is None else f'{dev:.4f}'}")
        ok = rms is not None and rms < a.rms_tol and abs(now - before) <= a.gl_tol_km and dev is not None and dev <= a.flux_tol
        stable = stable + 1 if ok else 0
        if stable >= a.stable_batches:
            log(f"BRANCH {tag}: converged ({stable} batches in a row): grounding line {now:.2f} km, "
                f"{now - xgl:+.2f} km from the S1 state's {xgl:.2f} km")
            break
    else:
        log(f"BRANCH {tag}: not converged in {chunks} chunks (last grounding line {gl_km(f['trans']):.2f} km)")
    out = subprocess.run([PY, "diag/ssa_diagnose.py", f["trans"], "--out", os.path.join(bdir, "diag")], cwd=ROOT,
                         capture_output=True, text=True)
    txt = (out.stdout + out.stderr).splitlines()
    k = next((i for i, l in enumerate(txt) if l.startswith("F.")), None)
    if k is not None:
        log(f"BRANCH {tag}: ssa_diagnose, flotation point:\n" + "\n".join(txt[k:k + 5]))
    open(os.path.join(bdir, "DONE"), "w").write(time.strftime("%F %T\n"))


if __name__ == "__main__":
    main()
