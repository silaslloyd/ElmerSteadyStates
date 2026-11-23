#!/usr/bin/env python3
"""A FREE transient from a converged steady state, run the ElmerStreams way: the grounding line by contact every
step, ElmerStreams' thermal chain every step ([elmer] thermal = "elmerstreams": the cold/temperate mask by
ColdTempMaskSolver, temperate -> cold only where the water sheet is gone; the sheet solved; friction heating from the
loads), heat and strain heating solved, the bedrock's real heat capacity, dt 5 yr. (--thermal frozen: the mask moved
only between chunks by ess/masks.py's heat-load test instead: it froze whole patches at once, 2026-11-23.) Does the state stay where Newton put it? In particular:
does the grounding line (1039.25 km on the whole sweep branch) stay, or does it go towards where the ElmerStreams
flowline run settles (714 km)?

    1. mesh <work>/F<head>/ (config <work>/refined/F<head>.toml): 250 m from --fine-from-km (default head - 20) to
       1065 km, so the grounding line is resolved wherever it goes between there and the shelf; 2 km upstream;
    2. ess setup / init, the state carried over (ess transfer), the shelf break (as every sweep state);
    3. ess run --gl contact --remask --dt DT --steps STEPS per chunk, chunks until the job ends, the state rewritten to
       <work>/free_transient/F<head>/run.npz after every chunk, kept every --save-every chunks in run_chunks/points/;
       resubmitting continues from run.npz.
--report: the grounding line, the patch head, the divide surface over the saved chunk states (and the latest run.npz).

    python3 jobs/free_transient.py CONFIG --from-state S.npz [--dt 5] [--steps 40] [--save-every 5] [--report]
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sweep_refined import write_config  # noqa: E402
from stability_spanwise import set_key  # noqa: E402
from ess.config import load_config  # noqa: E402
from ess.masks import _on  # noqa: E402
from ess.spectrum import _bed, gl_position  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} free-transient: {msg}", flush=True)


def run(cmd):
    log("$ " + " ".join(cmd))
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit("step failed")


def summary(path):
    """(time, grounding line km, patch head km, divide Zs m, temperate fraction of the grounded bed)"""
    s = load_state(path)
    g, xpos, grounded = _bed(s.frozen, s.diag)
    ct = _on(*s.frozen[find_name(s.frozen, "ColdTempMask")], g)
    xg = float(xpos[grounded].max()) if grounded.any() else np.nan
    o = np.argsort(-xpos)                       # from the grounding line upstream: the contiguous temperate patch
    head = np.nan
    for i in o:
        if not grounded[i]:
            continue
        if ct[i] > 0:
            head = xpos[i]
        else:
            break
    zs = s.layout.slices()
    bz = next(b for b in s.layout.blocks if b.name.lower() == "zs")
    xs = _on(*s.diag[find_name(s.diag, "coordinate 1")], bz.gids)
    zdiv = float(s.x[zs[bz.name]][np.argmin(xs)])
    # the temperate share of the grounded bed by LENGTH (node counts are skewed by the graded mesh)
    xg_, oo = xpos[grounded], np.argsort(xpos[grounded])
    xs_, ts_ = xg_[oo], (ct[grounded] > 0)[oo].astype(float)
    tf = float(np.sum(np.diff(xs_) * 0.5 * (ts_[1:] + ts_[:-1])) / (xs_[-1] - xs_[0])) if len(xs_) > 1 else np.nan
    t = float(s.meta.get("time", np.nan))
    return t, xg / 1e3, head / 1e3, zdiv, tf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--from-state", required=True)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--fine-from-km", type=float)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--front-km", type=float, help="the calving front here instead of the config's (e.g. 816.15: "
                    "ElmerStreams' flowline domain)")
    ap.add_argument("--no-shelf-break", action="store_true", help="the bed as the tables give it (ElmerStreams')")
    ap.add_argument("--tag", help="the run's name (default F<head>)")
    ap.add_argument("--thermal", choices=("elmerstreams", "frozen"), default="elmerstreams",
                    help="elmerstreams (default): ElmerStreams' thermal chain in Elmer, every step: ColdTempMaskSolver "
                         "(temperate -> cold only where the water sheet is gone, cold -> temperate above melting), the "
                         "water sheet solved, friction heating from the loads; frozen: the mask held in each chunk "
                         "and moved between chunks by ess/masks.py (temperate -> cold wherever the heat loads freeze: "
                         "NOT ElmerStreams' rule)")
    ap.add_argument("--sheet", choices=("cs", "elmerstreams"), default="cs",
                    help="the water sheet solver with --thermal elmerstreams (cs: SheetSolverhwCS)")
    ap.add_argument("--shelf-break", default="--strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 --width-km 10")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work = cfg["_work"]
    s0 = load_state(a.from_state)
    xgl = gl_position(s0.frozen, s0.diag) / 1e3
    t0, _, head0, _, _ = summary(a.from_state)
    tag = (a.tag or f"F{int(round(head0)):04d}") + ("e" if a.thermal == "elmerstreams" else "")
    odir = os.path.join(work, "free_transient", tag)
    out = os.path.join(odir, "run.npz")
    if a.report:
        files = sorted(glob.glob(os.path.join(odir, "run_chunks", "points", "state_*.npz")))
        print(f"{'state':>34s} {'t (yr)':>10s} {'GL (km)':>9s} {'head (km)':>10s} {'divide Zs':>10s} {'temperate':>9s}")
        for f in [a.from_state] + files + ([out] if os.path.exists(out) else []):
            t, g, h, z, tf = summary(f)
            print(f"{os.path.relpath(f, work)[-34:]:>34s} {t:10.0f} {g:9.2f} {h:10.2f} {z:10.1f} {tf:9.1%}")
        return
    x0 = a.fine_from_km if a.fine_from_km is not None else head0 - 20.0
    front = a.front_km or float(cfg["case"].get("front_km", cfg["case"]["res_points"][-1][0]))
    pts = ([[0.0, 2000.0], [x0 - 60.0, 2000.0], [x0, 250.0], [1065.0, 250.0], [1080.0, 1000.0], [front, 1000.0]]
           if front > 1085.0 else [[0.0, 2000.0], [x0 - 60.0, 2000.0], [x0, 250.0], [front, 250.0]])
    cfgf = os.path.join(work, "refined", f"{tag}.toml")
    os.makedirs(os.path.dirname(cfgf), exist_ok=True)
    write_config(a.config, cfgf, pts, "{work}/" + tag, None, None)
    import re
    txt = open(cfgf).read()
    if a.thermal == "elmerstreams":             # ElmerStreams' cold/temperate rule and water sheet, every step
        for k, v in (("thermal", '"elmerstreams"'), ("friction_heat", '"loads"'), ("strain_heat", "true"),
                     ("sheet_solver", f'"{a.sheet}"')):
            txt = set_key(txt, "elmer", k, v)
        txt, k = re.subn(r'^frozen\s*=\s*\[(.*?)\]', lambda m: m.group(0) if "Water Sheet" in m.group(1) else
                         f'frozen = [{m.group(1)}, "Water Sheet Thickness"]', txt, count=1, flags=re.M)
        if k != 1:
            raise SystemExit(f"{cfgf}: no [state] frozen line")
    open(cfgf, "w").write(txt)
    if a.front_km:
        txt = open(cfgf).read()
        txt, k = re.subn(r"^front_km\s*=.*$", f"front_km = {front:g}", txt, count=1, flags=re.M)
        if k != 1:
            raise SystemExit(f"{cfgf}: no front_km line")
        open(cfgf, "w").write(txt)
    log(f"source {a.from_state}: grounding line {xgl:.2f} km, patch head {head0:.2f} km; mesh 250 m from {x0:g} to "
        f"{min(front, 1065.0):g} km, front {front:g} km{', no shelf break' if a.no_shelf_break else ''}; dt {a.dt:g} yr x {a.steps} steps per chunk; config {cfgf}")
    mdir = os.path.join(work, tag)
    py = sys.executable
    st0, st1, st2 = (os.path.join(mdir, n) for n in ("state0.npz", "start0.npz", "start.npz"))
    if not os.path.exists(st0):
        run([py, "-u", "-m", "ess", "setup", cfgf])
        run([py, "-u", "-m", "ess", "init", cfgf, "--out", st0])
    if not os.path.exists(st1):
        run([py, "-u", "-m", "ess", "transfer", cfgf, "--state", st0, "--from", a.from_state, "--out", st1])
    if a.no_shelf_break:
        st2 = st1
    elif not os.path.exists(st2):
        run([py, "jobs/shelf_break.py", st1, st2] + a.shelf_break.split() + ["--force"])
    os.makedirs(odir, exist_ok=True)
    start, off, toff = st2, 0, 0.0
    if os.path.exists(out):
        start = out
        off = len(glob.glob(os.path.join(odir, "run_chunks", "points", "state_*.npz"))) * a.save_every
        toff = summary(out)[0]
        toff = 0.0 if not np.isfinite(toff) else toff
        log(f"continuing from {out} at t = {toff:g} yr (chunk offset {off})")
    run([py, "-u", "-m", "ess", "run", cfgf, "--state", start, "--gl", "contact",
         "--no-remask" if a.thermal == "elmerstreams" else "--remask", "--dt", f"{a.dt:g}",
         "--steps", str(a.steps), "--chunks", "100000", "--save-every", str(a.save_every), "--chunk-offset", str(off),
         "--time-offset", f"{toff:g}", "--out", out])


if __name__ == "__main__":
    main()
