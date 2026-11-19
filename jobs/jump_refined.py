#!/usr/bin/env python3
"""JUMP to a distant point of the marine bed sweep (a temperate patch of length d far from the last converged one),
without the small-step continuation: a transient with the masks fixed brings the state near the new steady state,
then Newton (with the contact grounding-line rounds, as the sweep) converges it.

The stability of the converged sweep points (jobs/flowline_stability.cmd, d = 794.2 km): with the patch and the
grounding line held, nothing grows; the slow modes are the bedrock's heat (hundreds of kyr), the ice's geometry ~14 kyr
and faster. The transient here uses Newton's map ([map]: the bedrock's heat capacity x 0.001, so its modes are gone in
well under 1 kyr; the steady states are the same), so it settles on the ice's time scales.

    1. the mesh: the base config's res_points with a FINE m band over [head - PAD, head + PAD] (graded over GRADE),
       config <work>/refined/j<head km>.toml, mesh <work>/j<head km>/ (as jobs/sweep_refined.py);
    2. ess setup, ess init, ess transfer --from the source state, jobs/shelf_break.py -> <work>/j<head>/start.npz;
    3. the patch for d (temperate within d of the grounding line, cold upstream), the masks fixed; the map
       x -> Phi(x) ([map], one Elmer run each) iterated, rms(Phi(x) - x) = rms(F) logged, every state saved in
       <work>/jump/j<head>/transient/; until rms(F) < SWITCH (default 3e-2: the size Newton took from the 12.5 km
       steps) or MAX_KYR (default 100 kyr);
    4. ess sweep --d-start-km d --d-end-km d on the transient's end: Newton and the contact rounds, the point in
       <work>/jump/j<head>/sweep/ (sweep.csv, points/state_XXXXX.npz).
Rerunning resumes: the mesh and start states kept, the transient from its last saved state, the Newton step as
ess sweep resumes.

    python3 jobs/jump_refined.py CONFIG (--head-km H | --d-km D) [--from-state S.npz | --from-refined DIR]
                                 [--pad-km 10] [--grade-km 10] [--fine 250] [--switch 3e-2] [--max-kyr 100]
                                 [--gl-rounds 6] [--transient-only]

--from-refined DIR (default <work>/sweep_refined): the converged point with the smallest d in DIR/seg_*/sweep.csv.
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sweep_refined import banded_points, converged, write_config  # noqa: E402
from ess.config import load_config  # noqa: E402
from ess.spectrum import gl_position  # noqa: E402
from ess.statefile import load_state  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} jump: {msg}", flush=True)


def run(cmd):
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--head-km", type=float)
    ap.add_argument("--d-km", type=float)
    ap.add_argument("--from-state")
    ap.add_argument("--from-refined")
    ap.add_argument("--pad-km", type=float, default=10.0)
    ap.add_argument("--grade-km", type=float, default=10.0)
    ap.add_argument("--fine", type=float, default=250.0)
    ap.add_argument("--switch", type=float, default=3e-2)
    ap.add_argument("--max-kyr", type=float, default=100.0)
    ap.add_argument("--gl-rounds", type=int, default=6)
    ap.add_argument("--transient-only", action="store_true")
    ap.add_argument("--shelf-break", default="--strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 --width-km 10")
    a = ap.parse_args()
    if (a.head_km is None) == (a.d_km is None):
        raise SystemExit("give one of --head-km or --d-km")
    cfg = load_config(a.config)
    work = cfg["_work"]
    c = cfg["case"]
    base = c["res_points"]
    front_km = float(c.get("front_km", base[-1][0]))
    # the source: a given state, or the most advanced converged point of the refined sweep
    if a.from_state:
        src = a.from_state
    else:
        root = a.from_refined or os.path.join(work, "sweep_refined")
        pts = [p for f in sorted(glob.glob(os.path.join(root, "seg_*"))) for p in converged(f)]
        if not pts:
            raise SystemExit(f"no converged point under {root}")
        src = min(pts)[1]
    s0 = load_state(src)
    xgl = gl_position(s0.frozen, s0.diag) / 1e3
    d = a.d_km if a.d_km is not None else round((xgl - a.head_km) * 10) / 10
    head = xgl - d
    tag = f"j{int(round(head)):04d}"
    log(f"from {os.path.relpath(src, ROOT)} (grounding line {xgl:.2f} km) to d = {d:g} km, head {head:.2f} km ({tag})")
    # 1-2. the mesh and the start state
    pts = banded_points(base, head - a.pad_km, head + a.pad_km, a.fine, a.grade_km, front_km)
    cfgf = os.path.join(work, "refined", f"{tag}.toml")
    mdir = os.path.join(work, tag)
    os.makedirs(os.path.dirname(cfgf), exist_ok=True)
    write_config(a.config, cfgf, pts, "{work}/" + tag)
    log(f"mesh: band {a.fine:g} m over {head - a.pad_km:.1f} .. {head + a.pad_km:.1f} km; res_points {pts}")
    py = sys.executable
    st0, st1, st2 = (os.path.join(mdir, f) for f in ("state0.npz", "start0.npz", "start.npz"))
    if not os.path.exists(st0):
        run([py, "-u", "-m", "ess", "setup", cfgf])
        run([py, "-u", "-m", "ess", "init", cfgf, "--out", st0])
    if not os.path.exists(st1):
        run([py, "-u", "-m", "ess", "transfer", cfgf, "--state", st0, "--from", src, "--out", st1])
    if not os.path.exists(st2):
        run([py, "jobs/shelf_break.py", st1, st2] + a.shelf_break.split() + ["--force"])
    # 3. the transient with the patch for d, the masks fixed
    jdir = os.path.join(work, "jump", tag)
    tdir = os.path.join(jdir, "transient")
    os.makedirs(tdir, exist_ok=True)
    end = os.path.join(jdir, "transient_end.npz")
    if not os.path.exists(end):
        from ess.cli import _patch_diagnostics, _problem, _setup
        from ess.model import rms
        from ess.spectrum import patch_mask
        from ess.statefile import find_name, save_state
        done = sorted(glob.glob(os.path.join(tdir, "state_*.npz")))
        start = done[-1] if done else st2
        cfg2, model, saved = _setup(SimpleNamespace(config=cfgf, state=start, verbose=False))
        _patch_diagnostics(model)
        prob = _problem(cfg2, model)
        kct = find_name(saved.frozen, "ColdTempMask")
        g_ct, ct, xg = patch_mask(saved.frozen, saved.diag, d * 1e3)
        saved.frozen[kct] = (g_ct, ct)
        T = prob.spec.horizon
        k0 = len(done)
        t = float(saved.meta.get("time", 0.0)) if done else 0.0
        nmax = int(np.ceil(a.max_kyr * 1e3 / T))
        log(f"transient: the patch for d = {d:g} km fixed ({int(np.sum(ct > 0))} temperate bed nodes), the grounding "
            f"line fixed; [map] {prob.spec.n_steps} x {prob.spec.dt:g} yr (bedrock heat capacity x "
            f"{prob.spec.rock_heat_capacity:g}) per chunk; until rms(F) < {a.switch:g} or {a.max_kyr:g} kyr"
            + (f"; resuming after chunk {k0} ({t:g} yr)" if done else ""))
        y = prob.scale(saved.x)
        r = np.inf
        for k in range(k0, nmax):
            yn = prob.phi(y, saved.p, tag=f"jump_{tag}_{k + 1}", want_diag=True)
            r = rms(yn - y)
            t += T
            y = yn
            save_state(os.path.join(tdir, f"state_{k + 1:04d}.npz"), saved.layout, prob.unscale(y), saved.p,
                       dict(saved.frozen), prob.last_diag, time=t, sweep_d=d * 1e3, x_gl=xg)
            log(f"transient: chunk {k + 1}  t = {t / 1e3:g} kyr  rms(F) = {r:.3e}")
            if r < a.switch:
                break
        save_state(end, saved.layout, prob.unscale(y), saved.p, dict(saved.frozen), prob.last_diag, time=t,
                   sweep_d=d * 1e3, x_gl=xg)
        log(f"transient: {'reached' if r < a.switch else 'NOT reached'} rms(F) < {a.switch:g} after {t / 1e3:g} kyr "
            f"(rms(F) {r:.3e}); -> {os.path.relpath(end, ROOT)}")
    if a.transient_only:
        return
    # 4. Newton with the contact rounds (one sweep point)
    run([py, "-u", "-m", "ess", "sweep", cfgf, "--state", end, "--d-start-km", f"{d:g}", "--d-end-km", f"{d:g}",
         "--gl-rounds", str(a.gl_rounds), "--out", os.path.join(jdir, "sweep")])
    log("done")


if __name__ == "__main__":
    main()
