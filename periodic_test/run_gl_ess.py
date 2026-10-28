#!/usr/bin/env python3
"""
Periodic grounding-line test on the ElmerStreams Marine_Term setup (README_gl_ess.md, configs/test_periodic_gl.toml).

The ESS marine flowline (bedrock, filler, ice, bed interface and surface bodies, the contact grounding line
moving every step) made several elements wide (configs/test_periodic_gl*.toml), side 2 the periodic image of
side 1, the sliding coefficient varied periodically across y so the grounding line is not straight; thermomechanical
(heat on, the cold/temperate mask moved between chunks). Started from a 2D marine state (ess transfer), then three
phases: spin-up until steady, retreat (accumulation x 0.3), advance (accumulation x 3) (PHASES). Each phase is `ess run --chunks N --save-every 1`,
so there is a state after every chunk. For each periodic_gl treatment:
    elmerstreams   ElmerStreams' contact.f90 (raw loads, Fw doubled on the master side: Fw2)
    fix            SlidCoef_ContactPeriodic + the grounded mask made periodic after GroundedSolver
    none           Elmer/Ice SlidCoef_Contact (raw loads, Fw undoubled)

    python3 periodic_test/run_gl_ess.py prep [--from STATE]        setup, init, transfer (to <work>/start.npz)
    python3 periodic_test/run_gl_ess.py run MODE                   the three phases for one treatment
    python3 periodic_test/run_gl_ess.py analyse [MODE ...]         per chunk: grounding line per row across y,
                                                                   largest master/image mismatch of each field
Usually through periodic_test/run_gl_ess.sh (batch jobs).
"""
import argparse
import csv
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
# GL_CONFIG (environment) picks the config, e.g. configs/test_periodic_gl_50km.toml; the work dir is its work_dir
CONFIG = os.environ.get("GL_CONFIG", os.path.join("configs", "test_periodic_gl.toml"))


def _work():
    from ess.config import load_config
    return load_config(os.path.join(REPO, CONFIG))["_work"]


WORK = _work()
START_FROM = os.path.join("work", "marine_flowline", "run_cold_p268.15.npz")
MODES = ("elmerstreams", "fix", "none")
# (name, chunks, fcoeff factor, aDIM factor): chunks of [map] n_steps x dt (100 yr)
# Thermomechanical (heat on, cold/temperate mask moved between chunks, made periodic; the grounding line by contact).
# (name, chunks, fcoeff factor, aDIM factor, save every N chunks, stop when the rms change per chunk < tol (None:
# run all chunks)). The spin-up runs until steady (the Stokes-only runs never got there in 500 yr), then the forcing:
# accumulation x 0.3 (retreat), x 3 (advance). Friction factors did nothing on a cold bed; --fbase is still there.
# Earlier (Stokes only, --bed temperate): fbase 1 ran the grounding line to the calving front, 10..100 advanced it.
PHASES = [("spinup", 100, 1.0, 1.0, 5, 2e-3), ("retreat", 10, 1.0, 0.3, 1, None), ("advance", 10, 1.0, 3.0, 1, None)]


def ess(*args):
    cmd = [sys.executable, "-m", "ess", *args]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def base_params():
    from ess.case import eval_lua_parameters
    P = eval_lua_parameters(os.path.join(REPO, "case", "parameters.lua"))
    return float(P["fcoeff"]), float(P["aDIM"])


def cmd_prep(a):
    ess("setup", CONFIG)
    ess("init", CONFIG)
    ess("transfer", CONFIG, "--from", a.source, "--out", os.path.join(WORK, "start.npz"))


def cmd_run(a):
    fc, ad = base_params()
    state = os.path.join(WORK, "start.npz")
    tag = a.tag or (a.mode if a.fbase == 1.0 else f"{a.mode}_f{a.fbase:g}")
    resume = None
    if a.resume:                                    # PHASE: from that phase's last saved chunk (or the phase before)
        names = [p[0] for p in PHASES]
        if a.resume not in names:
            raise SystemExit(f"--resume: one of {names}")
        k = names.index(a.resume)
        done = sorted(glob.glob(os.path.join(WORK, tag, f"{a.resume}_chunks", "points", "state_*.npz")))
        if a.resume_at is not None:                 # a given chunk instead of the last one
            done = [f for f in done if int(os.path.basename(f)[6:10]) <= a.resume_at]
        if done:
            state, resume = done[-1], (a.resume, int(os.path.basename(done[-1])[6:10]))
        elif k > 0:
            state = os.path.join(WORK, tag, f"{names[k - 1]}.npz")
        print(f"resume {a.resume}: from {state}" + (f" (after chunk {resume[1]})" if resume else ""), flush=True)
    os.makedirs(os.path.join(WORK, tag), exist_ok=True)
    print(f"run {a.mode}: fcoeff x {a.fbase:g} (x the phase factors), results in {os.path.join(WORK, tag)}", flush=True)
    phases = PHASES[:1] if a.spinup_only else PHASES
    if a.resume:
        phases = phases[[p[0] for p in phases].index(a.resume):]
    for name, chunks, ff, af, every, tol in phases:
        if name == "spinup":
            chunks = a.spinup_max or chunks
            tol = a.spinup_tol if a.spinup_tol is not None else tol
        out = os.path.join(WORK, tag, f"{name}.npz")
        n = a.chunks or chunks
        offset = 0
        if resume and resume[0] == name:
            offset = resume[1]
            n -= offset
            if n <= 0:
                state = out
                continue
        args = ["run", CONFIG, "--state", state, "--out", out, "--gl", "contact",
                "--periodic-gl", a.mode, "--chunks", str(n), "--save-every", str(every), "--chunk-offset", str(offset),
                "--lua", f"fcoeff={fc * a.fbase * ff!r}", "--lua", f"aDIM={ad * af!r}"]
        if tol:
            args += ["--until-tol", repr(tol)]
        if a.stokes_only:       # no heat: Stokes and free surface, and no remask (temperature frozen)
            args += ["--stokes-only", "--no-remask"]
        else:                   # the cold/temperate mask follows the temperature (periodic); Elmer moves the grounding line
            args += ["--remask"]
        if a.bed:
            args += ["--bed", a.bed]
        ess(*args)
        state = out
    cmd_analyse(argparse.Namespace(modes=[tag]))         # the table at the end of the job


# ---------------------------------------------------------------- analysis
def _field(rec, name):
    from ess.statefile import find_name
    try:
        k = find_name(rec, name)
    except KeyError:
        return None
    return rec[k]


def analyse_state(path, width):
    from ess.statefile import load_state
    s = load_state(path)
    allf = dict(s.diag)
    allf.update(s.frozen)                         # the grounded mask Elmer ended the chunk with (contact)
    blocks = s.layout.split(s.x)
    for b in s.layout.blocks:
        allf[b.name] = (b.gids, blocks[b.name])
    gx = _field(allf, "coordinate 1")
    gy = _field(allf, "coordinate 2")
    gz = _field(allf, "coordinate 3")
    if gx is None:
        raise SystemExit(f"{path}: no coordinates in the diagnostics")
    gids, X = gx
    Y = dict(zip(*gy))
    Z = dict(zip(*gz))
    # columns on the two sides, nodes ranked by z within each column: (x, rank) pairs master (y = 0) and image
    cols = {}
    for g, x in zip(gids.tolist(), X.tolist()):
        y = Y[g]
        side = 0 if abs(y) < 1 else (1 if abs(y - width) < 1 else None)
        if side is None:
            continue
        cols.setdefault((round(x, 1), side), []).append((Z[g], g))
    key = {}
    for (xr, side), lst in cols.items():
        for r, (_, g) in enumerate(sorted(lst)):
            key[g] = (xr, r, side)

    def mism(name, comps=None, scale=None):
        names = [name] if comps is None else [f"{name} {c}" for c in comps]
        recs = [_field(allf, n) for n in names]
        if any(r is None for r in recs):
            return np.nan
        side = ({}, {})
        for ci, (g, v) in enumerate(recs):
            for gi, vi in zip(g.tolist(), v.tolist()):
                k = key.get(gi)
                if k is not None:
                    side[k[2]].setdefault(k[:2], [np.nan] * len(recs))[ci] = vi
        common = side[0].keys() & side[1].keys()
        if not common:
            return np.nan
        d = max(max(abs(a - b) for a, b in zip(side[0][k], side[1][k])) for k in common)
        return d / (scale or 1.0)
    u = [_field(allf, f"Velocity {c}") for c in (1, 2, 3)]
    umax = max(float(np.max(np.abs(r[1]))) for r in u if r is not None) if any(r is not None for r in u) else 1.0
    gm = _field(allf, "GroundedMask")
    lz = _field(allf, "Flow Solution Loads 3")
    out = dict(mask=mism("GroundedMask"), Zs=mism("Zs"), Zb=mism("Zb"),
               du=mism("Velocity", (1, 2, 3), umax),
               loads_z=mism("Flow Solution Loads 3", scale=float(np.max(np.abs(lz[1]))) if lz is not None else 1.0),
               t=float(s.meta.get("time", np.nan)))
    gl = []
    if gm is not None:
        Xd = dict(zip(gids.tolist(), X.tolist()))
        xm = np.array([Xd[g] for g in gm[0].tolist()])
        ym = np.array([Y[g] for g in gm[0].tolist()])
        for yj in np.unique(np.round(ym, 0)):
            m = (np.abs(ym - yj) < 1) & (gm[1] > -0.5)
            gl.append(float(xm[m].max()) / 1e3 if m.any() else np.nan)
    out["GL"] = gl
    return out


def cmd_analyse(a):
    from ess.config import load_config
    width = float(load_config(os.path.join(REPO, CONFIG))["case"]["width"])
    modes = a.modes or sorted(d for d in os.listdir(WORK) if os.path.isdir(os.path.join(WORK, d))
                              and any(os.path.isdir(os.path.join(WORK, d, f"{p[0]}_chunks")) for p in PHASES))
    for mode in modes:
        files = []
        for name, *_ in PHASES:
            files += [(name, f) for f in sorted(glob.glob(os.path.join(WORK, mode, f"{name}_chunks", "points",
                                                                        "state_*.npz")))]
        if not files:
            print(f"\n{mode}: no chunk states yet ({os.path.join(WORK, mode)})")
            continue
        print(f"\n{mode}: grounding line (last grounded x, km) per row across y (y = 0 and y = W are the same row),"
              f" and the largest master (y = 0) / image (y = W) mismatch after each chunk")
        print(f"{'phase':8s} {'chunk':>5s} {'GL rows y = 0 .. W (km)':>44s} {'mask':>5s} {'Zs (m)':>9s} "
              f"{'Zb (m)':>9s} {'|du|/u':>9s} {'loads z':>9s}")
        rows = []
        for name, f in files:
            r = analyse_state(f, width)
            r["phase"], r["chunk"] = name, int(os.path.basename(f)[6:10])
            rows.append(r)
            fmt = lambda v: f"{v:9.2g}" if np.isfinite(v) else f"{'-':>9s}"
            g_ = np.array(r["GL"], float)
            gls = (" ".join(f"{v:7.2f}" for v in g_) if len(g_) <= 6 else
                   f"min {np.nanmin(g_):7.2f} mean {np.nanmean(g_):7.2f} max {np.nanmax(g_):7.2f}")
            print(f"{name:8s} {r['chunk']:5d} {gls:>44s} {r['mask']:5.0f} {fmt(r['Zs'])} {fmt(r['Zb'])} "
                  f"{fmt(r['du'])} {fmt(r['loads_z'])}")
        with open(os.path.join(WORK, mode, "periodicity.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            n = max(len(r["GL"]) for r in rows)
            w.writerow(["phase", "chunk", "mask", "Zs", "Zb", "du", "loads_z"] + [f"GL_row{j}" for j in range(n)])
            for r in rows:
                w.writerow([r[k] for k in ("phase", "chunk", "mask", "Zs", "Zb", "du", "loads_z")]
                           + r["GL"] + [np.nan] * (n - len(r["GL"])))
        gl0 = np.array([r["GL"][0] if r["GL"] else np.nan for r in rows])
        worst = {k: np.nanmax([r[k] for r in rows]) for k in ("mask", "Zs", "Zb", "du")}
        if np.isfinite(gl0).any():
            print(f"  grounding line at y = 0: {np.nanmin(gl0):.2f} .. {np.nanmax(gl0):.2f} km")
        print("  worst master/image mismatch: " + ", ".join(f"{k} {v:.2g}" for k, v in worst.items())
              + "   (mask: 0, 1 or 2; |du|/u: velocity / max speed; loads z: raw Stokes loads, not periodic)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep")
    s.add_argument("--from", dest="source", default=START_FROM, help=f"2D marine state (default {START_FROM})")
    s.set_defaults(fn=cmd_prep)
    s = sub.add_parser("run")
    s.add_argument("mode", choices=MODES)
    s.add_argument("--chunks", type=int, help="chunks per phase (default per phase: "
                                              + ", ".join(f"{p[0]} {p[1]}" for p in PHASES) + ")")
    s.add_argument("--fbase", type=float, default=1.0, help="sliding coefficient fcoeff times this in every phase")
    s.add_argument("--stokes-only", action="store_true", help="no heat solve (temperature frozen, no remask)")
    s.add_argument("--bed", choices=["temperate", "cold"], help="the whole bed temperate or cold (ess run --bed)")
    s.add_argument("--spinup-tol", type=float, help=f"spin-up until the rms change per chunk is below this "
                                                    f"(default {PHASES[0][5]:g}; 0: all chunks)")
    s.add_argument("--spinup-max", type=int, help=f"at most this many spin-up chunks (default {PHASES[0][1]})")
    s.add_argument("--spinup-only", action="store_true", help="only the spin-up phase (to choose --fbase)")
    s.add_argument("--tag", help="results folder under the work dir (default MODE, or MODE_f<FBASE>)")
    s.add_argument("--resume", metavar="PHASE", help="continue from PHASE's last saved chunk (then the later phases)")
    s.add_argument("--resume-at", type=int, metavar="N", help="with --resume: from chunk N of that phase, not the last")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("analyse")
    s.add_argument("modes", nargs="*", help="results folders (treatments, e.g. fix, fix_f30; default all)")
    s.set_defaults(fn=cmd_analyse)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
