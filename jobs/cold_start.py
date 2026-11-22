#!/usr/bin/env python3
"""The subtemperate (all-cold bed) steady state at the run's surface temperature, reached from a frozen-bed ice sheet
by warming the surface, as the start of a patch sweep UP from the grounding line (d = 0 -> d_end):

    0. its own mesh (<work>/cold_mesh, config <work>/refined/cold.toml: the base resolution), SRC carried onto it
       (default the coarse sweep's last converged point), the shelf break as every sweep segment's
                                                                               -> <work>/cold/start.npz
    1. spin-up, mechanics only, bed frozen (ESS_SLIDING=frozen: no sliding where grounded; the contact grounding line
       moves every step), at the cold surface temperature TC (default 258.15 K), from SRC, in runs of --spin-batch chunks of the [map]
       (10 x 2000 yr) until the rms change per chunk < --spin-tol (1e-2)    -> <work>/cold/spin.npz, spin.done
    2. Newton on the geometry, mechanics only, bed frozen (ess newton --stokes-only; the grounding line by contact
       between solves)                                                         -> <work>/cold/frozen.npz
    3. the steady temperature on that geometry at TC (ess run --heat-only, --heat-steps x --heat-dt yr, the bed still
       frozen: no friction heat, the cold branch), the whole grounded bed COLD (--bed cold: the model's flux
       condition)                                                              -> <work>/cold/heat.npz
       The grounded bed's warmest temperature is printed: subtemperate sliding is frozen for practical purposes
       below T0 - 2.07 K (gamma ln 1e6, the friction cap).
    4. full Newton at TC (Stokes + heat, subtemperate sliding, the bed all cold, the grounding line by contact)
                                                                               -> <work>/cold/cold_p<TC>.npz
    5. pseudo-arclength continuation in the surface temperature from TC up to TEND (default 268.15; [continuation]
       of the config, p_max = TEND, max_folds 1: a fold of the cold branch before TEND ends it, reported), the bed all
       cold, the grounding line by contact after each point                   -> <work>/branch_cold/
    6. Newton at exactly TEND from the branch's last point below it (before any fold)
                                                                               -> <work>/cold/cold_p<TEND>.npz
Then (not here) the patch sweep up from it: ess sweep CONFIG --state <work>/cold/cold_p268.15.npz --d-start-km 0
--d-end-km D. Every step is skipped when its output exists (resubmitting resumes; the continuation resumes itself).

    python3 jobs/cold_start.py CONFIG [--from-state SRC] [--tc 258.15] [--tend 268.15] [--spin-batch 10] [--spin-tol 1e-2]
                               [--heat-dt 1e4] [--heat-steps 30] [--stop-after N]
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sweep_refined import write_config  # noqa: E402
from ess.config import load_config  # noqa: E402
from ess.spectrum import _bed, basal_temperature  # noqa: E402
from ess.statefile import load_state  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} cold: {msg}", flush=True)


def run(cmd, env=None, check=True):
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if check and r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode})")
    return r.returncode


def bed_report(path):
    s = load_state(path)
    g, xpos, gr = _bed(s.frozen, s.diag)
    T = basal_temperature(s.layout, s.x, g) - 273.15
    Tg = T[gr]
    k = int(np.argmax(Tg))
    n2 = int((Tg > -2.07).sum())
    log(f"{os.path.basename(path)}: grounded bed T - T0 from {Tg.min():+.2f} to {Tg.max():+.2f} K (warmest at "
        f"x = {xpos[gr][k] / 1e3:.1f} km); {n2} of {len(Tg)} grounded bed nodes warmer than T0 - 2.07 K (sliding "
        f"above the friction cap there), grounding line {xpos[gr].max() / 1e3:.2f} km")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--from-state")
    ap.add_argument("--tc", type=float, default=258.15)
    ap.add_argument("--tend", type=float, default=268.15)
    ap.add_argument("--spin-batch", type=int, default=10, help="spin-up chunks per run (2000 yr each with the [map])")
    ap.add_argument("--spin-tol", type=float, default=1e-2, help="the spin-up stops when the rms change per chunk is below")
    ap.add_argument("--heat-dt", type=float, default=1e4)
    ap.add_argument("--heat-steps", type=int, default=30)
    ap.add_argument("--shelf-break", default="--strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 --width-km 10")
    ap.add_argument("--stop-after", type=int, default=6, help="stop after this step (1-6)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work = cfg["_work"]
    cdir = os.path.join(work, "cold")
    os.makedirs(cdir, exist_ok=True)
    src = a.from_state
    if not src:                     # the coarse sweep's last converged point (this config's own mesh)
        f = os.path.join(work, "sweep_p268.15_down", "sweep.csv")
        rows = [r for r in csv.DictReader(open(f)) if str(r.get("converged")) == "1"]
        src = os.path.join(work, "sweep_p268.15_down", "points",
                           f"state_{int(round(float(rows[-1]['d_km']) * 10)):05d}.npz")
    py = sys.executable
    tc = f"{a.tc:g}"
    # its own mesh (the base config's resolution, partitioned for this job's ranks), the source carried onto it
    base_cfg = a.config
    a.config = os.path.join(work, "refined", "cold.toml")
    os.makedirs(os.path.dirname(a.config), exist_ok=True)
    write_config(base_cfg, a.config, cfg["case"]["res_points"], "{work}/cold_mesh")
    mdir = os.path.join(work, "cold_mesh")
    st0, st1, st2 = os.path.join(mdir, "state0.npz"), os.path.join(mdir, "start0.npz"), os.path.join(cdir, "start.npz")
    if not os.path.exists(st0):
        run([py, "-u", "-m", "ess", "setup", a.config])
        run([py, "-u", "-m", "ess", "init", a.config, "--out", st0])
    if not os.path.exists(st1):
        run([py, "-u", "-m", "ess", "transfer", a.config, "--state", st0, "--from", src, "--out", st1])
    if not os.path.exists(st2):                     # the shelf break, as every sweep segment's start
        run([py, "jobs/shelf_break.py", st1, st2] + a.shelf_break.split() + ["--force"])
    src_name, src = src, st2
    frozen = dict(os.environ, ESS_SLIDING="frozen")
    log(f"from {os.path.relpath(src_name, ROOT)} (on the mesh {mdir}); cold surface {tc} K, warming to {a.tend:g} K; into {cdir}")
    spin, fz, heat = (os.path.join(cdir, f) for f in ("spin.npz", "frozen.npz", "heat.npz"))
    cold = os.path.join(cdir, f"cold_p{tc}.npz")
    # the spin-up runs until the rms change per chunk < --spin-tol (no resubmitting by hand): batches of --spin-batch
    # chunks, each from the last saved spin.npz (a job ended by its time limit resumes from there); spin.done once
    # converged. Later steps' outputs made before it converged are stale (renamed .old).
    done_f = os.path.join(cdir, "spin.done")
    if not os.path.exists(done_f):
        for f in (fz, heat, cold):
            if os.path.exists(f):
                os.replace(f, f + ".old")
        tmp = os.path.join(cdir, "spin_tmp.npz")
        total = 0
        while True:
            st_in = spin if os.path.exists(spin) else src
            log(f"spin-up: {a.spin_batch} chunks from {os.path.basename(st_in)}, stopping once the rms change per "
                f"chunk < {a.spin_tol:g}")
            log("$ ess run ... (frozen bed, mechanics only)")
            pr = subprocess.Popen([py, "-u", "-m", "ess", "run", a.config, "--state", st_in, "--p", tc, "--stokes-only",
                                   "--bed", "cold", "--chunks", str(a.spin_batch), "--until-tol", f"{a.spin_tol:g}",
                                   "--out", tmp], cwd=ROOT, env=frozen, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
            ok, last = False, None
            for line in pr.stdout:
                print(line, end="", flush=True)
                m = re.search(r"rms change ([0-9.eE+-]+)", line)
                if m:
                    last = float(m.group(1))
                if "rms change" in line and "<" in line and "over one chunk" in line:
                    ok = True
            if pr.wait() != 0:
                raise SystemExit(f"spin-up run failed (exit {pr.returncode})")
            os.replace(tmp, spin)
            total += a.spin_batch
            if ok or (last is not None and last < a.spin_tol):
                open(done_f, "w").write(f"rms change {last}\n")
                log(f"spin-up converged: rms change per chunk {last:.3e} < {a.spin_tol:g}")
                break
            log(f"spin-up: rms change per chunk still {last:.3e} after this batch; going on")
    bed_report(spin)
    if a.stop_after <= 1:
        return
    if not os.path.exists(fz):
        rc = run([py, "-u", "-m", "ess", "newton", a.config, "--state", spin, "--p", tc, "--stokes-only", "--out", fz],
                 env=frozen, check=False)
        if rc != 0:
            log("the frozen-bed Newton did not converge: going on from its last state (the heat solve and the full "
                "Newton follow)")
    if a.stop_after <= 2:
        return
    if not os.path.exists(heat):
        env = dict(frozen, ESS_SEA_SPRING_DT=os.environ.get("ESS_SEA_SPRING_DT", "50"))
        run([py, "-u", "-m", "ess", "run", a.config, "--state", fz, "--p", tc, "--heat-only", "--dt", f"{a.heat_dt:g}",
             "--steps", str(a.heat_steps), "--gl", "frozen", "--no-remask", "--bed", "cold", "--out", heat], env=env)
    bed_report(heat)
    if a.stop_after <= 3:
        return
    if not os.path.exists(cold):
        run([py, "-u", "-m", "ess", "newton", a.config, "--state", heat, "--p", tc, "--out", cold])
    bed_report(cold)
    if a.stop_after <= 4:
        return
    # 5. the continuation: a config copy with [continuation] name cold, p_max TEND, max_folds 1
    ccfg = os.path.join(cdir, "cold_continuation.toml")
    text = open(a.config).read()                   # (the cold mesh's config)
    sec = re.search(r"^\[continuation\]\s*$(.*?)(?=^\[|\Z)", text, flags=re.M | re.S)
    if not sec:
        raise SystemExit(f"{a.config}: no [continuation] section")
    body = sec.group(1)
    for k, v in (("name", '"cold"'), ("p_max", f"{a.tend:g}"), ("max_folds", "1")):
        body, n = re.subn(rf"^{k}\s*=.*$", f"{k} = {v}", body, count=1, flags=re.M)
        if n == 0:
            body = f"{k} = {v}\n" + body
    text = text[:sec.start(1)] + body + text[sec.end(1):]
    with open(ccfg, "w") as f:
        f.write(text)
    bdir = os.path.join(work, "branch_cold")
    run([py, "-u", "-m", "ess", "continue", ccfg, "--state", cold, "--p", tc], check=False)
    if a.stop_after <= 5:
        return
    # 6. Newton at exactly TEND from the last point below it, before any fold
    out = os.path.join(cdir, f"cold_p{a.tend:g}.npz")
    if os.path.exists(out):
        bed_report(out)
        return
    rows = list(csv.DictReader(open(os.path.join(bdir, "branch.csv"))))
    best = None
    for k, r in enumerate(rows):
        if str(r.get("fold", "")) not in ("", "0", "False"):
            log(f"the cold branch folds at point {k} (p = {float(r['p']):.4g} K)" +
                ("" if float(r["p"]) >= a.tend else f": before {a.tend:g} K, no cold state there on this branch"))
            break
        if float(r["p"]) <= a.tend + 1e-9:
            best = k
    if best is None:
        raise SystemExit("no branch point below TEND")
    st = os.path.join(bdir, "points", f"state_{best:04d}.npz")
    log(f"Newton at {a.tend:g} K from branch point {best} (p = {float(rows[best]['p']):.4g} K)")
    run([py, "-u", "-m", "ess", "newton", ccfg, "--state", st, "--p", f"{a.tend:g}", "--out", out])
    bed_report(out)


if __name__ == "__main__":
    main()
