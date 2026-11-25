#!/usr/bin/env python3
"""gl_newton_gl.py

Newton with the grounding line as an unknown. `ess newton --stokes-only` finds the steady state for the grounding line it
is given (the mask is frozen), and the grounding line of a marine flowline on a nearly flat bed is almost neutral (it relaxes
over ~100 kyr), so the steady position is found here as the root of a scalar function of the imposed position x_g:

    r(x_g) = x_f - x_g,   x_f = the flotation point of the converged steady state (ess/glroot.py)

Each evaluation = one frozen-mask Newton (the pipeline's S2 solve, ESS_SLIDING etc. from the environment, as
jobs/gl_newton.cmd) at the imposed x_g, then r from the converged state. The positions are chosen by a first step of
--step0-km in the direction r points, then a secant method (inside a bracket once r has changed sign); the next start state is
the last converged state with its grounded mask moved to the new x_g (ess/masks.py update_masks, gl_x). The remask loop
of `ess newton` is switched off in a copy of the config ([masks] auto = false), so the imposed position is not moved.

    python3 jobs/gl_newton_gl.py --config work/gl_ssa/refined/branch_ssa125.toml --from work/gl_ssa/branch_ssa125/transient.npz
    jobs/gl_newton_gl.cmd runs it as a job (TAG, FROM).

Writes to <work>/glroot_<tag>/: start_KK.npz / steady_KK.npz per evaluation KK, history.json (resumable: finished
evaluations are not redone), RESULT.txt. Stops when |r| <= --r-tol (m) (the position of the root is then known to about
r-tol / |dr/dx_g|, the slope being ~0.003 on a nearly flat bed: 10 m is ~3 km) or the bracket is narrower than --w-tol-km, or
after --max-evals evaluations. --no-newton: only the residual of --from and the move to the first trial position (no Elmer).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from ess import glroot  # noqa: E402
from ess.case import eval_lua_parameters  # noqa: E402
from ess.config import load_config  # noqa: E402
from ess.statefile import load_state, save_state  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} glroot: {msg}", flush=True)


def set_key(text, section, key, value):
    """`key = value` in [section] of a toml text (added if absent, the section too)"""
    m = re.search(rf"^\[{re.escape(section)}\]\s*$", text, re.M)
    if not m:
        return text.rstrip() + f"\n\n[{section}]\n{key} = {value}\n"
    nxt = re.search(r"^\[", text[m.end():], re.M)
    end = m.end() + nxt.start() if nxt else len(text)
    sec = text[m.end():end]
    if re.search(rf"^{re.escape(key)}\s*=", sec, re.M):
        sec = re.sub(rf"^{re.escape(key)}\s*=.*$", f"{key} = {value}", sec, count=1, flags=re.M)
    else:
        sec = f"\n{key} = {value}" + sec
    return text[:m.end()] + sec + text[end:]


def fine_range_m(cfg):
    """the 250 m-or-finer band of the mesh (m), from the config's res_points: the positions the grounding line may take"""
    pts = cfg.get("case", {}).get("res_points", [])
    fine = [float(x) for x, res in pts if float(res) <= 250.0]
    return (min(fine) * 1e3, max(fine) * 1e3) if fine else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the branch's config (work/<case>/refined/branch_<tag>.toml)")
    ap.add_argument("--from", dest="src", required=True, help="the start state (a state of the branch)")
    ap.add_argument("--tag", default="glroot")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--step0-km", type=float, default=10.0, help="the first step, in the direction r points")
    ap.add_argument("--max-step-km", type=float, default=25.0)
    ap.add_argument("--xmin-km", type=float, default=None, help="default: 10 km inside the fine band of the mesh")
    ap.add_argument("--xmax-km", type=float, default=None)
    ap.add_argument("--r-tol", type=float, default=10.0, help="m")
    ap.add_argument("--w-tol-km", type=float, default=1.0)
    ap.add_argument("--max-evals", type=int, default=8)
    ap.add_argument("--maxit", type=int, default=None, help="[newton] maxit in the config copy")
    ap.add_argument("--retries", type=int, default=2, help="a Newton that does not converge: halve the step, this many times")
    ap.add_argument("--no-newton", action="store_true")
    ap.add_argument("--newton-cmd", default=None,
                    help="TESTING: a command replacing `python3 -m ess newton` (called as CMD CONFIG --state S --out OUT)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    P = eval_lua_parameters(cfg["_path"](cfg.get("case", {}).get("parameters", "case/parameters.lua")))
    rho = float(P["rhoi"] / P["rhow"])
    odir = a.out_dir or os.path.join(cfg["_work"], f"glroot_{a.tag}")
    os.makedirs(odir, exist_ok=True)
    fr = fine_range_m(cfg)
    xmin = a.xmin_km * 1e3 if a.xmin_km is not None else (fr[0] + 10e3 if fr else 0.0)
    xmax = a.xmax_km * 1e3 if a.xmax_km is not None else (fr[1] - 10e3 if fr else 1e9)
    log(f"rho_i/rho_w {rho:.5f}; grounding line allowed in [{xmin / 1e3:.1f}, {xmax / 1e3:.1f}] km; out {odir}")

    text = open(a.config).read()
    text = set_key(text, "masks", "auto", "false")
    if a.maxit:
        text = set_key(text, "newton", "maxit", str(a.maxit))
    cfgp = os.path.join(odir, "newton.toml")
    open(cfgp, "w").write(text)

    hpath = os.path.join(odir, "history.json")
    hist = json.load(open(hpath)) if os.path.exists(hpath) else []
    if hist:
        log(f"resuming: {len(hist)} evaluation(s) done")
    path = lambda kind, k: os.path.join(odir, f"{kind}_{k:02d}.npz")
    if not os.path.exists(path("start", 0)) and not hist:
        shutil.copyfile(a.src, path("start", 0))

    k, retries = len(hist), 0
    while k < a.max_evals:
        start, out = path("start", k), path("steady", k)
        if not os.path.exists(start):
            log(f"evaluation {k}: no {start}")
            break
        if not os.path.exists(out) and not a.no_newton:
            log(f"evaluation {k}: Newton from {os.path.basename(start)}")
            t0 = time.time()
            if a.newton_cmd:
                import shlex
                cmd = shlex.split(a.newton_cmd) + [cfgp, "--state", start, "--out", out]
            else:
                cmd = [sys.executable, "-u", "-m", "ess", "newton", cfgp, "--state", start, "--stokes-only", "--out", out]
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            log(f"evaluation {k}: Newton exit {rc} after {(time.time() - t0) / 60:.1f} min")
            if rc != 0:
                if k == 0 or not hist or retries >= a.retries:
                    log("Newton did not converge" + (" at the start state" if k == 0 else f" after {retries} retries")
                        + "; stop (raise [newton] maxit with --maxit, or start from a state closer to steady)")
                    break
                good = load_state(path("steady", len(hist) - 1))
                xg_good = glroot.residual(good, rho)["x_gl"]
                xg_try = glroot.residual(load_state(start), rho)["x_gl"] if os.path.exists(start) else xg_good
                x_half = 0.5 * (xg_good + xg_try)
                log(f"retry {retries + 1}: halve the step, x_g {xg_try / 1e3:.2f} -> {x_half / 1e3:.2f} km")
                xn, frz, rep = glroot.shift_gl(good, x_half, rho)
                save_state(start, good.layout, xn, good.p, frz, good.diag)
                os.remove(out) if os.path.exists(out) else None
                retries += 1
                continue
        src = out if os.path.exists(out) else start
        saved = load_state(src)
        res = glroot.residual(saved, rho)
        rec = dict(k=k, x_g=res["x_gl"], x_f=res["x_f"], r=res["r"], state=src, n_below_bed=res["n_base_below_bed"],
                   below_bed_max=res["base_below_bed_max"], H_gl=res.get("H_last_grounded"), Hf_gl=res.get("Hf_last_grounded"))
        hist.append(rec)
        json.dump(hist, open(hpath, "w"), indent=1)
        retries = 0
        log(f"evaluation {k}: x_g {res['x_gl'] / 1e3:.3f} km, flotation point x_f {res['x_f'] / 1e3:.3f} km, "
            f"r = {res['r']:+.1f} m;  H at the last grounded column {res.get('H_last_grounded', float('nan')):.2f} m vs "
            f"flotation {res.get('Hf_last_grounded', float('nan')):.2f} m; floating base below the bed: "
            f"{res['n_base_below_bed']} columns, max {res['base_below_bed_max']:.2f} m")
        pts = [(h["x_g"], h["r"]) for h in hist if np.isfinite(h["r"])]
        if not pts:
            log("no flotation point in this state: the thickness never crosses flotation; stop")
            break
        if abs(res["r"]) <= a.r_tol:
            log(f"|r| = {abs(res['r']):.1f} m <= {a.r_tol:g} m: converged, the steady grounding line is {res['x_gl'] / 1e3:.2f} km")
            break
        xs_pos = [x for x, r in pts if r > 0]
        xs_neg = [x for x, r in pts if r < 0]
        if xs_pos and xs_neg:                                   # r has changed sign: the narrowest bracket
            width = min(abs(p - q) for p in xs_pos for q in xs_neg)
            if width <= a.w_tol_km * 1e3:
                log(f"bracket {width / 1e3:.2f} km <= {a.w_tol_km:g} km: stop")
                break
        if a.no_newton:
            xn_try, why = glroot.next_x(pts, xmin, xmax, a.step0_km * 1e3, a.max_step_km * 1e3)
            log(f"--no-newton: next position {xn_try / 1e3:.2f} km ({why})")
            xn, frz, rep = glroot.shift_gl(saved, xn_try, rho)
            save_state(path("start", k + 1), saved.layout, xn, saved.p, frz, saved.diag)
            log(f"wrote {path('start', k + 1)} (grounding line moved: {rep.get('to_grounded', 0)} columns' nodes grounded, "
                f"{rep.get('to_floating', 0)} floated)")
            break
        xn_try, why = glroot.next_x(pts, xmin, xmax, a.step0_km * 1e3, a.max_step_km * 1e3)
        if abs(xn_try - res["x_gl"]) < 1.0:
            log(f"the next position {xn_try / 1e3:.2f} km is where we are (the edge of the allowed range?): stop")
            break
        log(f"next: x_g {xn_try / 1e3:.2f} km ({why})")
        xn, frz, rep = glroot.shift_gl(saved, xn_try, rho)
        save_state(path("start", k + 1), saved.layout, xn, saved.p, frz, saved.diag)
        k += 1

    pts = [(h["x_g"], h["r"]) for h in hist if np.isfinite(h["r"])]
    est = glroot.root_estimate(pts)
    best = min(hist, key=lambda h: abs(h["r"]) if np.isfinite(h["r"]) else np.inf) if hist else None
    lines = [f"{len(hist)} evaluation(s)"] + [f"  {h['k']}: x_g {h['x_g'] / 1e3:.3f} km  r {h['r']:+.1f} m  ({os.path.basename(h['state'])})"
                                             for h in hist]
    if best:
        lines.append(f"smallest |r|: x_g {best['x_g'] / 1e3:.3f} km, r {best['r']:+.1f} m, state {best['state']}")
    if est is not None:
        lines.append(f"root of r from the two points nearest r = 0: {est / 1e3:.2f} km")
    open(os.path.join(odir, "RESULT.txt"), "w").write("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
