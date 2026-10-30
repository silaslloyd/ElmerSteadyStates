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
    if os.environ.get("GL_WORK"):           # GL_WORK=DIR: results somewhere else (e.g. slimmed copies, for analyse)
        return os.path.abspath(os.environ["GL_WORK"])
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
    if a.source == "none":                          # the initial state of the config's own tables is the start
        ess("init", CONFIG, "--out", os.path.join(WORK, "start.npz"))
        return
    ess("init", CONFIG)
    ess("transfer", CONFIG, "--from", a.source, "--out", os.path.join(WORK, "start.npz"))


def cmd_run(a):
    fc, ad = base_params()
    state = a.start or os.path.join(WORK, "start.npz")
    tag = a.tag or (f"{a.mode}_{a.heat}" + ("" if a.fbase == 10.0 else f"_f{a.fbase:g}")
                    + (f"_{a.precond.lower()}" if a.precond else ""))
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
    print(f"run {a.mode}, friction heating {a.heat}: fcoeff x {a.fbase:g} (x the phase factors), bed {a.bed}, "
          f"results in {os.path.join(WORK, tag)}", flush=True)
    phases = PHASES[:1] if a.spinup_only else PHASES
    sea = {p[0]: None for p in phases}             # sea level per phase (m; None: the config's)
    if a.phase:                                     # --phase NAME=CHUNKS,FRICTION,ACCUMULATION[,SEALEVEL]
        table = {p[0]: list(p) for p in phases}
        for spec in a.phase:
            try:
                name, vals = spec.split("=")
                v = vals.split(",")
                table[name][1:4] = [int(v[0]), float(v[1]), float(v[2])]
                if len(v) > 3:
                    sea[name] = float(v[3])
                elif len(v) != 3:
                    raise ValueError
            except (ValueError, KeyError, IndexError):
                raise SystemExit(f"--phase {spec!r}: NAME=CHUNKS,FRICTION,ACCUMULATION[,SEALEVEL] with NAME one of "
                                 f"{', '.join(table)}")
        phases = [tuple(table[p[0]]) for p in phases]
        for p_ in phases:
            print(f"  phase {p_[0]}: {p_[1]} chunks, fcoeff x {p_[2]:g}, accumulation x {p_[3]:g}"
                  + (f", sea level {sea[p_[0]]:g} m" if sea[p_[0]] is not None else ""), flush=True)
    if a.resume:
        phases = phases[[p[0] for p in phases].index(a.resume):]
    for name, chunks, ff, af, every, tol in phases:
        if name == "spinup":
            chunks = a.spinup_max or chunks
            tol = a.spinup_tol if a.spinup_tol is not None else tol
        out = os.path.join(WORK, tag, f"{name}.npz")
        n = a.chunks or chunks
        every = a.save_every or every
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
        if sea.get(name) is not None:               # needs [elmer] sea_level_param = true
            args += ["--lua", f"ess_sl={sea[name]!r}"]
        if tol:
            args += ["--until-tol", repr(tol)]
        # Elmer moves the grounding line (contact) and, with [elmer] thermal = "elmerstreams", the cold/temperate
        # mask and the water sheet every step: no ESS remask between chunks
        args += ["--stokes-preconditioner", a.precond] if a.precond else []
        args += ["--linear-stokes", a.linear_stokes] if a.linear_stokes else []
        args += ["--stokes-element", a.stokes_element] if a.stokes_element else []
        args += ["--stokes-block-file", a.stokes_block_file] if a.stokes_block_file else []
        args += ["--no-remask", "--friction-heat", a.heat] + (["--stokes-only"] if a.stokes_only else [])
        if a.bed != "none":
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
    plz = _field(allf, "Periodic Flow Solution Loads 3")
    out = dict(mask=mism("GroundedMask"), Zs=mism("Zs"), Zb=mism("Zb"),
               ctmask=mism("ColdTempMask"),
               hw=mism("Periodic Flow Solution Loads 3", scale=float(np.max(np.abs(plz[1]))) if plz is not None else 1.0),
               T=mism("Temperature"),
               du=mism("Velocity", (1, 2, 3), umax),
               loads_z=mism("Flow Solution Loads 3", scale=float(np.max(np.abs(lz[1]))) if lz is not None else 1.0),
               t=float(s.meta.get("time", np.nan)))
    gl = []
    if gm is not None:
        Xd = dict(zip(gids.tolist(), X.tolist()))
        xm = np.array([Xd[g] for g in gm[0].tolist()])
        ym = np.array([Y[g] for g in gm[0].tolist()])
        for yj in np.unique(np.round(ym, 0)):
            m = np.abs(ym - yj) < 1
            gl.append(gl_upstream(xm[m], gm[1][m]) / 1e3 if m.any() else np.nan)
    out["GL"] = gl
    out["Tseam"], out["Tint"] = seam_stripe(bed_rows(allf, gids, X, Y, width))
    out["Zseam"], out["Zint"] = seam_stripe(field_rows(allf, "Zs", gids, X, Y, width))
    return out


def field_rows(allf, name, gids, X, Y, width):
    """a surface field: {x: [value per row, y = 0 .. the last row before W]} where every row has a node"""
    F = _field(allf, name)
    if F is None:
        return {}, []
    Xd = dict(zip(gids.tolist(), X.tolist()))
    rows = {}
    for g, v in zip(F[0].tolist(), F[1].tolist()):
        if Y[g] > width - 1:
            continue
        rows.setdefault(round(Xd[g], 1), {})[round(Y[g], 0)] = v
    ys = sorted({y for d in rows.values() for y in d})
    return {x: [d[y] for y in ys] for x, d in rows.items() if len(d) == len(ys)}, ys


def gl_upstream(x, mask):
    """the grounding line as the upstream edge of the shelf: the last grounded node (mask >= 0) before the first
    floating one (mask -1) going downstream, so grounded patches under the shelf do not count (all grounded: the
    last node)"""
    o = np.argsort(x)
    x, m = np.asarray(x)[o], np.asarray(mask)[o]
    fl = np.where(m < -0.5)[0]
    if len(fl) == 0:
        return float(x[-1])
    k = fl[0] - 1
    return float(x[k]) if k >= 0 else np.nan


def bed_rows(allf, gids, X, Y, width):
    """bed temperature on the grounded bed: {x: [T per row, y = 0 .. the last row before W]} (y = W is y = 0)"""
    gm, T = _field(allf, "GroundedMask"), _field(allf, "Temperature")
    if gm is None or T is None:
        return {}, []
    Td = dict(zip(*T))
    Xd = dict(zip(gids.tolist(), X.tolist()))
    ys = sorted({round(Y[g], 0) for g in gm[0].tolist() if Y[g] < width - 1})
    rows = {}
    for g, m in zip(gm[0].tolist(), gm[1].tolist()):
        if Y[g] > width - 1 or g not in Td:
            continue
        rows.setdefault(round(Xd[g], 1), {})[round(Y[g], 0)] = (Td[g], m)
    out = {}
    for x, d in rows.items():
        if len(d) == len(ys) and all(m > -0.5 for _, m in d.values()):
            out[x] = [d[y][0] for y in ys]
    return out, ys


def seam_stripe(br):
    """largest |T_j - (T_j-1 + T_j+1) / 2| over the grounded bed on the seam row (j = 0, wrapping) and on the middle
    row (y = W/2: like the seam, where the sliding bump sin(2 pi y / W) is zero; the other rows if no middle row)"""
    rows, ys = br
    if not rows or len(ys) < 3:
        return np.nan, np.nan
    A = np.array(list(rows.values()))                    # (x, row)
    d2 = A - 0.5 * (np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1))
    n = len(ys)
    ref = d2[:, n // 2] if n % 2 == 0 else d2[:, 1:]
    return float(np.max(np.abs(d2[:, 0]))), float(np.max(np.abs(ref)))


def cmd_analyse(a):
    from ess.config import load_config
    width = _width()
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
        print(f"\n{mode}: grounding line (km, upstream edge of the shelf) per row across y (y = 0 and y = W are the same row),"
              f" and the largest master (y = 0) / image (y = W) mismatch after each chunk")
        print(f"{'phase':8s} {'chunk':>5s} {'GL rows y = 0 .. W (km)':>44s} {'mask':>5s} {'ct':>3s} {'Zs (m)':>9s} "
              f"{'Zb (m)':>9s} {'|du|/u':>9s} {'T (K)':>9s} {'ploads z':>9s} {'loads z':>9s} {'Tb seam':>9s} {'Tb mid':>9s} {'Zs seam':>9s} {'Zs mid':>9s}")
        rows = []
        for name, f in files:
            r = analyse_state(f, width)
            r["phase"], r["chunk"] = name, int(os.path.basename(f)[6:10])
            rows.append(r)
            fmt = lambda v: f"{v:9.2g}" if np.isfinite(v) else f"{'-':>9s}"
            g_ = np.array(r["GL"], float)
            gls = (" ".join(f"{v:7.2f}" for v in g_) if len(g_) <= 6 else
                   f"min {np.nanmin(g_):7.2f} mean {np.nanmean(g_):7.2f} max {np.nanmax(g_):7.2f}")
            ct = f"{r['ctmask']:3.0f}" if np.isfinite(r["ctmask"]) else "  -"
            print(f"{name:8s} {r['chunk']:5d} {gls:>44s} {r['mask']:5.0f} {ct} {fmt(r['Zs'])} {fmt(r['Zb'])} "
                  f"{fmt(r['du'])} {fmt(r['T'])} {fmt(r['hw'])} {fmt(r['loads_z'])} {fmt(r['Tseam'])} {fmt(r['Tint'])} {fmt(r['Zseam'])} {fmt(r['Zint'])}")
        with open(os.path.join(WORK, mode, "periodicity.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            n = max(len(r["GL"]) for r in rows)
            keys = ("phase", "chunk", "mask", "ctmask", "Zs", "Zb", "du", "T", "ploads_z", "loads_z", "Tseam", "Tint", "Zseam",
                    "Zint")
            for r in rows:
                r["ploads_z"] = r["hw"]
            w.writerow(list(keys) + [f"GL_row{j}" for j in range(n)])
            for r in rows:
                w.writerow([r[k] for k in keys]
                           + r["GL"] + [np.nan] * (n - len(r["GL"])))
        gl0 = np.array([r["GL"][0] if r["GL"] else np.nan for r in rows])
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            worst = {k: np.nanmax([r[k] for r in rows]) for k in ("mask", "ctmask", "Zs", "Zb", "du", "T", "ploads_z",
                                                                  "Tseam", "Tint", "Zseam", "Zint")}
        if np.isfinite(gl0).any():
            print(f"  grounding line at y = 0: {np.nanmin(gl0):.2f} .. {np.nanmax(gl0):.2f} km")
        print("  worst master/image mismatch: " + ", ".join(f"{k} {v:.2g}" for k, v in worst.items())
              + "   (mask / ct: grounded / cold-temperate mask, 0, 1 or 2; |du|/u: velocity / max speed; hw: water sheet"
              " / max; ploads z / loads z: the periodic copy / the raw Stokes loads (not periodic), / max)\n"
              "  Tb seam / Tb mid: largest |T_j - (T_j-1 + T_j+1)/2| of the grounded bed temperature across y (K) on the "
              "seam row / on the middle row y = W/2 (where the sliding bump is zero too): a seam stripe (wrong seam heating) shows "
              "as Tb seam >> Tb mid\n  Zs seam / Zs mid: the same for the surface elevation (m): a ridge or trough along the seam")


def _width():
    """the strip width across the flow: [case] width, else the mesh default Wmesh of parameters.lua"""
    from ess.config import load_config
    from ess.case import eval_lua_parameters
    c = load_config(os.path.join(REPO, CONFIG))["case"]
    if "width" in c:
        return float(c["width"])
    return float(eval_lua_parameters(os.path.join(REPO, "case", "parameters.lua")).get("Wmesh", 1000.0))


def cmd_solvers(a):
    """the same run with different Stokes linear solvers (jobs/gl_solvers.sh): Elmer time per phase and per step,
    and how far each run's grounding line and surface are from the reference (the first) run's"""
    from ess.statefile import load_state
    tags = a.tags or [t for t in ("direct", "iterative", "block", "amg") if os.path.isdir(os.path.join(WORK, t))]
    if not tags:
        raise SystemExit(f"no solver results in {WORK}")
    ref = tags[0]
    rev = sorted(glob.glob(os.path.join(WORK, "runs", "*", "elmer.log")))
    if rev:
        with open(rev[-1], errors="replace") as fh:
            for line in fh:
                if "Version" in line or "Rev:" in line or "Compiled" in line:
                    print("Elmer: " + line.strip())
                    break
    # time per phase (the phase outputs record the Elmer wall time of their ess run; resumed phases only the rest)
    print(f"\nElmer wall time (s) and per time step; speed-up = reference time / this run's time (reference {ref})")
    print(f"{'phase':8s} " + " ".join(f"{t:>22s}" for t in tags))
    tot = {t: [0.0, 0] for t in tags}
    for name, *_ in PHASES:
        cells = []
        for t in tags:
            f = os.path.join(WORK, t, f"{name}.npz")
            if not os.path.exists(f):
                cells.append(f"{'-':>22s}")
                continue
            m = dict(load_state(f).meta)
            sec, st = float(m.get("elmer_seconds", np.nan)), int(m.get("steps", 0))
            tot[t][0] += sec
            tot[t][1] += st
            cells.append(f"{sec:9.0f} {sec / max(st, 1):7.2f}/st x{st:<4d}")
        print(f"{name:8s} " + " ".join(cells))
    r0 = tot[ref][0] / max(tot[ref][1], 1)
    print(f"{'total':8s} " + " ".join(f"{v[0]:9.0f} {v[0] / max(v[1], 1):7.2f}/st x{v[1]:<4d}" for v in tot.values()))
    print(f"{'speed-up':8s} " + " ".join(f"{r0 / (v[0] / v[1]) if v[1] and v[0] else np.nan:22.2f}" for v in tot.values()))

    # grounding line and surface along the way, against the reference
    def chunks(t):
        out = {}
        for name, *_ in PHASES:
            for f in sorted(glob.glob(os.path.join(WORK, t, f"{name}_chunks", "points", "state_*.npz"))):
                out[(name, int(os.path.basename(f)[6:10]))] = f
        return out

    def gl_zs(f):
        s = load_state(f)
        allf = dict(s.diag)
        allf.update(s.frozen)
        blocks = s.layout.split(s.x)
        for b in s.layout.blocks:
            allf[b.name] = (b.gids, blocks[b.name])
        g, X = _field(allf, "coordinate 1")
        Xd = dict(zip(g.tolist(), X.tolist()))
        gm = _field(allf, "GroundedMask")
        Yd = dict(zip(*_field(allf, "coordinate 2")))
        row = np.array([abs(Yd[i]) < 1.0 for i in gm[0].tolist()])
        xg = np.array([Xd[i] for i in gm[0].tolist()])
        gl = gl_upstream(xg[row], gm[1][row]) / 1e3
        zs = _field(allf, "Zs")
        return gl, dict(zip(zs[0].tolist(), zs[1].tolist()))
    C = {t: chunks(t) for t in tags}
    keys = [k for k in C[ref] if all(k in C[t] for t in tags)]
    order = {n: i for i, (n, *_) in enumerate(PHASES)}
    keys.sort(key=lambda k: (order[k[0]], k[1]))
    if not keys:
        print("\nno chunk states common to all runs yet")
        return
    print(f"\ngrounding line (km, upstream edge of the shelf, y = 0) after each saved chunk; dZs: largest |Zs - Zs({ref})| (m)")
    print(f"{'phase':8s} {'chunk':>5s} " + " ".join(f"{t:>10s}" for t in tags) + " "
          + " ".join(f"{'dZs ' + t:>14s}" for t in tags[1:]))
    worst = {t: [0.0, 0.0] for t in tags[1:]}
    for k in keys:
        R = {t: gl_zs(C[t][k]) for t in tags}
        dz = []
        for t in tags[1:]:
            common = R[ref][1].keys() & R[t][1].keys()
            d = max(abs(R[ref][1][g] - R[t][1][g]) for g in common)
            dz.append(d)
            worst[t][0] = max(worst[t][0], abs(R[t][0] - R[ref][0]))
            worst[t][1] = max(worst[t][1], d)
        print(f"{k[0]:8s} {k[1]:5d} " + " ".join(f"{R[t][0]:10.2f}" for t in tags) + " "
              + " ".join(f"{d:14.3g}" for d in dz))
    print("largest difference from " + ref + ": "
          + "; ".join(f"{t}: grounding line {w[0]:.2f} km, Zs {w[1]:.3g} m" for t, w in worst.items()))


def cmd_compare(a):
    """bed temperature of run A minus run B (same mesh), per phase end and saved chunk: on the seam row and on the
    other rows. Loads vs flux heating: the same physics, so A - B is the discretisation difference, smooth across y;
    wrong loads on the seam show as a seam-row difference larger than on the other rows."""
    from ess.config import load_config
    from ess.statefile import load_state
    width = _width()

    def bed(path):
        s = load_state(path)
        allf = dict(s.diag)
        allf.update(s.frozen)
        blocks = s.layout.split(s.x)
        for b in s.layout.blocks:
            allf[b.name] = (b.gids, blocks[b.name])
        gx, gy = _field(allf, "coordinate 1"), _field(allf, "coordinate 2")
        return bed_rows(allf, gx[0], gx[1], dict(zip(*gy)), width)
    print(f"bed temperature {a.a} - {a.b} (K) on the grounded bed (where grounded in both), largest |difference| on "
          f"the seam row (y = 0 = W), the middle row (y = W/2) and all other rows, and the mean")
    print(f"{'phase':8s} {'chunk':>5s} {'seam':>8s} {'mid':>8s} {'others':>8s} {'mean':>8s} {'nx':>5s}")
    for name, *_ in PHASES:
        for fa in sorted(glob.glob(os.path.join(WORK, a.a, f"{name}_chunks", "points", "state_*.npz"))):
            fb = os.path.join(WORK, a.b, os.path.relpath(fa, os.path.join(WORK, a.a)))
            if not os.path.exists(fb):
                continue
            (ra, ya), (rb, yb) = bed(fa), bed(fb)
            xs = sorted(ra.keys() & rb.keys())
            if not xs or ya != yb:
                continue
            D = np.array([ra[x] for x in xs]) - np.array([rb[x] for x in xs])
            mid = np.max(np.abs(D[:, len(ya) // 2])) if len(ya) % 2 == 0 else np.nan
            print(f"{name:8s} {int(os.path.basename(fa)[6:10]):5d} {np.max(np.abs(D[:, 0])):8.3f} {mid:8.3f} "
                  f"{np.max(np.abs(D[:, 1:])):8.3f} {np.mean(D):8.3f} {len(xs):5d}")


def _state_fields(path):
    from ess.statefile import load_state
    s = load_state(path)
    allf = dict(s.diag)
    allf.update(s.frozen)
    blocks = s.layout.split(s.x)
    for b in s.layout.blocks:
        allf[b.name] = (b.gids, blocks[b.name])
    return allf


def cmd_glpos(a):
    """the grounding line of a state (km, the y = 0 row), for scripts"""
    allf = _state_fields(a.state)
    g, X = _field(allf, "coordinate 1")
    Xd = dict(zip(g.tolist(), X.tolist()))
    gm = _field(allf, "GroundedMask")
    Yd = dict(zip(*_field(allf, "coordinate 2")))
    row = np.array([abs(Yd[i]) < 1.0 for i in gm[0].tolist()])
    xg = np.array([Xd[i] for i in gm[0].tolist()])
    print(f"{gl_upstream(xg[row], gm[1][row]) / 1e3:.3f}")


def cmd_cutcheck(a):
    """a run on the shortened mesh ([case] front_km) against the same run on the whole domain: the surface speed
    along the y = 0 row, compared at the same x (the meshes have different node numbers)"""
    def surf(path):
        allf = _state_fields(path)
        g, X = _field(allf, "coordinate 1")
        Y = dict(zip(*_field(allf, "coordinate 2")))
        Z = dict(zip(*_field(allf, "coordinate 3")))
        u = _field(allf, "Velocity 1")
        if u is None:
            raise SystemExit(f"{path}: no Velocity 1 in the diagnostics")
        Xd = dict(zip(g.tolist(), X.tolist()))
        top = {}
        for i, v in zip(u[0].tolist(), u[1].tolist()):
            if i in Xd and abs(Y[i]) < 1.0:
                xr = round(Xd[i], 1)
                if xr not in top or Z[i] > top[xr][0]:
                    top[xr] = (Z[i], v)
        xs = np.array(sorted(top))
        return xs, np.array([top[x][1] for x in xs])
    xr, ur = surf(a.ref)
    xc, uc = surf(a.cut)
    m = xc < a.cut_km * 1e3 - 1.0
    d = np.abs(uc[m] - np.interp(xc[m], xr, ur))
    up = xc[m] < a.gl_km * 1e3
    near = np.abs(xc[m] - a.gl_km * 1e3) < 5e3
    ugl = np.interp(a.gl_km * 1e3, xr, ur)
    if getattr(a, "profile", False):          # the two surface-speed profiles near the grounding line, every ~1 km
        print("      x (km)   u ref    u test   (surface, y = 0)")
        for x in np.arange(a.gl_km - 10.0, a.cut_km + 0.01, 1.0):
            if x * 1e3 > xc.max():
                break
            print(f"      {x:7.1f} {np.interp(x * 1e3, xr, ur):8.2f} {np.interp(x * 1e3, xc, uc):8.2f}")
    print(f"      surface speed vs the whole domain: grounded ice max |du| {d[up].max():.3g} m/a "
          f"(max speed {np.abs(ur[xr < a.gl_km * 1e3]).max():.4g}), within 5 km of the grounding line max |du| "
          f"{d[near].max():.3g} m/a (speed there {ugl:.4g}), whole cut domain {d.max():.3g} m/a")


def _columns(path):
    """per mesh column (x, y rounded to 0.1 m): the nodes' horizontal velocity, the bed node's GroundedMask; and the
    grounding line per y row (km)"""
    allf = _state_fields(path)
    g, X = _field(allf, "coordinate 1")
    Xd = dict(zip(g.tolist(), X.tolist()))
    Yd = dict(zip(*_field(allf, "coordinate 2")))
    u1, u2 = _field(allf, "Velocity 1"), _field(allf, "Velocity 2")
    if u1 is None or u2 is None:
        raise SystemExit(f"{path}: no Velocity 1/2 in the diagnostics")
    U2 = dict(zip(*u2))
    vel = {}
    for i, v in zip(u1[0].tolist(), u1[1].tolist()):
        if i in Xd:
            vel.setdefault((round(Xd[i], 1), round(Yd[i], 1)), {})[i] = (v, U2.get(i, 0.0))
    gm = _field(allf, "GroundedMask")
    mask = {}
    for i, m in zip(gm[0].tolist(), gm[1].tolist()):
        if i in Xd:
            mask[(round(Xd[i], 1), round(Yd[i], 1))] = m
    rows = {}
    for (x, y), m in mask.items():
        rows.setdefault(y, ([], []))
        rows[y][0].append(x); rows[y][1].append(m)
    gl = {y: gl_upstream(np.array(xs), np.array(ms)) / 1e3 for y, (xs, ms) in rows.items()}
    return vel, mask, gl


def cmd_wavycheck(a):
    """two runs on the SAME mesh (e.g. the whole shelf and the moving full-Stokes region): the grounding line per y
    row, and the horizontal velocity difference (every node of a column, max) on the grounded ice, within 5 km of
    the local grounding line, and on the floating ice within the full-Stokes region (--d km beyond the grounding line)"""
    vr, mr, glr = _columns(a.ref)
    vt, mt, glt = _columns(a.test)
    ys = sorted(set(glr) & set(glt))
    gr = np.array([glr[y] for y in ys]); gt = np.array([glt[y] for y in ys])
    if a.rows:
        print("      y (km)   GL ref   GL test  (km)")
        for y, p, q in zip(ys, gr, gt):
            print(f"      {y/1e3:6.1f} {p:9.3f} {q:9.3f}")
    print(f"      grounding line (ref): {np.nanmin(gr):.2f} .. {np.nanmax(gr):.2f} km across y "
          f"(waviness {np.nanmax(gr) - np.nanmin(gr):.2f} km); |GL test - GL ref| max {np.nanmax(np.abs(gt - gr)):.3f} km, "
          f"mean {np.nanmean(np.abs(gt - gr)):.3f} km")
    glrow = dict(zip(ys, gr))
    yk = np.array(ys)
    dmax = {"grounded": 0.0, "near": 0.0, "float": 0.0}
    umax = 0.0
    for key, nodes in vr.items():
        if key not in vt:
            continue
        x, y = key
        j = int(np.argmin(np.abs(yk - y)))
        dist = x / 1e3 - glrow[ys[j]]                     # km downstream of the row's grounding line
        du = 0.0
        for i, (u, v) in nodes.items():
            if i in vt[key]:
                p, q = vt[key][i]
                du = max(du, float(np.hypot(p - u, q - v)))
                umax = max(umax, float(np.hypot(u, v)))
        if mr.get(key, -1.0) > -0.5 and dist <= 0.0:
            dmax["grounded"] = max(dmax["grounded"], du)
        if abs(dist) <= 5.0:
            dmax["near"] = max(dmax["near"], du)
        if 0.0 < dist <= a.d:
            dmax["float"] = max(dmax["float"], du)
    print(f"      horizontal velocity, max |du| (any depth): grounded ice {dmax['grounded']:.3g} m/a, within 5 km of the "
          f"local grounding line {dmax['near']:.3g} m/a, floating ice up to {a.d:g} km beyond it {dmax['float']:.3g} m/a "
          f"(max speed {umax:.4g} m/a)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep")
    s.add_argument("--from", dest="source", default=START_FROM, help=f"2D marine state (default {START_FROM}); "
                                                                       "none: the config's own initial state")
    s.set_defaults(fn=cmd_prep)
    s = sub.add_parser("run")
    s.add_argument("mode", choices=MODES)
    s.add_argument("--chunks", type=int, help="chunks per phase (default per phase: "
                                              + ", ".join(f"{p[0]} {p[1]}" for p in PHASES) + ")")
    s.add_argument("--start", help="start state (default <work dir>/start.npz)")
    s.add_argument("--heat", choices=["loads", "flux"], default="loads",
                   help="frictional heating: from the Stokes loads (nodal, the periodic copy with fix) or as a flux")
    s.add_argument("--fbase", type=float, default=10.0, help="sliding coefficient fcoeff times this in every phase "
                                                             "(default 10: keeps the grounding line inside)")
    s.add_argument("--stokes-only", action="store_true", help="no heat solve (temperature frozen, no remask)")
    s.add_argument("--bed", choices=["temperate", "cold", "none"], default="cold",
                   help="the cold/temperate mask over the whole bed (ess run --bed; default cold: the grounded bed "
                        "temperature free, no cap); none: the mask of the start state")
    s.add_argument("--spinup-tol", type=float, help=f"spin-up until the rms change per chunk is below this "
                                                    f"(default {PHASES[0][5]:g}; 0: all chunks)")
    s.add_argument("--spinup-max", type=int, help=f"at most this many spin-up chunks (default {PHASES[0][1]})")
    s.add_argument("--precond", help="Stokes preconditioner (e.g. ILU0); results in MODE_HEAT_<precond>")
    s.add_argument("--linear-stokes", choices=["direct", "iterative", "block"], help="Stokes linear solver (ess run)")
    s.add_argument("--stokes-block-file", help="block-preconditioner keywords (ess run; e.g. "
                                               "elmer/stokes_block_amg.sif.in)")
    s.add_argument("--stokes-element", choices=["P2/P1", "Stabilized", "Bubbles"],
                   help="Stokes element (ess run; default [elmer] stokes_element)")
    s.add_argument("--phase", action="append", metavar="NAME=CHUNKS,FRIC,ACC[,SEA]",
                   help="override a phase: chunks, friction factor, accumulation factor and optionally sea level in m "
                        "(needs [elmer] sea_level_param; e.g. retreat=50,1,1,50); "
                        "repeatable. Defaults: " + "; ".join(f"{p[0]}={p[1]},{p[2]:g},{p[3]:g}" for p in PHASES))
    s.add_argument("--save-every", type=int, help="save a state every N chunks in every phase (default per phase)")
    s.add_argument("--spinup-only", action="store_true", help="only the spin-up phase (to choose --fbase)")
    s.add_argument("--tag", help="results folder under the work dir (default MODE_HEAT, or MODE_HEAT_f<FBASE>)")
    s.add_argument("--resume", metavar="PHASE", help="continue from PHASE's last saved chunk (then the later phases)")
    s.add_argument("--resume-at", type=int, metavar="N", help="with --resume: from chunk N of that phase, not the last")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("analyse")
    s.add_argument("modes", nargs="*", help="results folders (treatments, e.g. fix, fix_f30; default all)")
    s.set_defaults(fn=cmd_analyse)
    s = sub.add_parser("solvers")
    s.add_argument("tags", nargs="*", help="results folders, the first the reference (default: direct iterative block "
                                           "amg, those present)")
    s.set_defaults(fn=cmd_solvers)
    s = sub.add_parser("glpos")
    s.add_argument("state")
    s.set_defaults(fn=cmd_glpos)
    s = sub.add_parser("cutcheck")
    s.add_argument("ref")
    s.add_argument("cut")
    s.add_argument("cut_km", type=float)
    s.add_argument("gl_km", type=float)
    s.add_argument("--profile", action="store_true", help="also print both surface-speed profiles near the grounding line")
    s.set_defaults(fn=cmd_cutcheck)
    s = sub.add_parser("wavycheck", help="two runs on the same mesh: grounding line per row, velocity differences")
    s.add_argument("ref")
    s.add_argument("test")
    s.add_argument("--d", type=float, default=10.0, help="km: the full-Stokes region beyond the grounding line")
    s.add_argument("--rows", action="store_true", help="print the grounding line of every y row")
    s.set_defaults(fn=cmd_wavycheck)
    s = sub.add_parser("compare")
    s.add_argument("a", nargs="?", default="fix_loads")
    s.add_argument("b", nargs="?", default="fix_flux")
    s.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
