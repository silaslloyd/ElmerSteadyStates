#!/usr/bin/env python3
"""The spreading test with the mesh following the current's margin (ess/spread_remesh.py): chunks of the transient
(STEPS x DT yr each, Elmer runs as `ess run`), and after every chunk the margin checked against the mesh; when it has
left its fine band, a new mesh (a<NNN> in the case dir) and the state carried over, then on.

    python3 jobs/spread_adapt.py CONFIG --state START.npz [--t0 T] --chunks N --dt DT --steps S [--out DIR]
                                 [--remesh-first] [--lua NAME=VALUE ...]

States after every chunk: DIR/points/state_NNNN.npz (meta time, mesh_db), DIR default <work>/adapt. The meshes and
their size views: <case dir>/a001, a001.geo, a001_size.pos, ... Resubmitting with --state the last saved state (its
mesh_db) goes on. --remesh-first: a new mesh fitted to the start state before the first chunk (a start on the uniform
setup mesh).
"""
import argparse
import glob
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} adapt: {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--state", required=True)
    ap.add_argument("--t0", type=float)
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--remesh-first", action="store_true")
    ap.add_argument("--lua", action="append", default=[])
    a = ap.parse_args()
    from ess.cli import _problem, _setup
    from ess.config import build_model
    from ess.model import MapSpec, rms
    from ess.spread_remesh import default_opt, needs_remesh, remesh_spread
    from ess.statefile import load_state, save_state

    cfg, model, saved = _setup(SimpleNamespace(config=a.config, state=a.state, verbose=False))
    case = cfg["paths"]["case_dir"]
    out = a.out or os.path.join(cfg["_work"], "adapt")
    pdir = os.path.join(out, "points")
    os.makedirs(pdir, exist_ok=True)
    opt = default_opt(cfg["case"])
    mesh = str(saved.meta.get("mesh_db", "") or cfg["elmer"].get("mesh_db", "footprint"))
    t = float(a.t0 if a.t0 is not None else saved.meta.get("time", 0.0))
    k0 = len(glob.glob(os.path.join(pdir, "state_*.npz")))
    n_procs = model.n_procs
    log(f"start {a.state} on mesh {mesh}, t = {t:g} yr; {a.chunks} chunks of {a.steps} x {a.dt:g} yr; remesh: fine "
        f"{opt['fine']:g} m within {opt['band'] / 1e3:g} km of the margin (H = {opt['Hc']:g} m), when the mesh at the "
        f"margin is coarser than {opt['trigger']:g} x fine; {n_procs} ranks")

    def next_name():
        k = 1
        while os.path.exists(os.path.join(case, f"a{k:03d}")):
            k += 1
        return f"a{k:03d}"

    def build(s, m):
        mdl = build_model(cfg, s)
        mdl.mesh_db = m
        for kv in a.lua:
            k_, _, v_ = kv.partition("=")
            mdl.lua_overrides[k_.strip()] = v_.strip()
        pr = _problem(cfg, mdl)
        pr.spec = MapSpec(dt=a.dt, n_steps=a.steps, coupled_iters=pr.spec.coupled_iters, rock_heat_capacity=1.0)
        return mdl, pr

    def remesh_now(s, why):
        nonlocal mesh
        name = next_name()
        log(f"remesh ({why}): {mesh} -> {name}")
        layout, x, frozen, diag, rep = remesh_spread(cfg, s, name, mesh, n_procs, opt)
        st = os.path.join(out, f"{name}_start.npz")
        save_state(st, layout, x, s.p, frozen, diag, mesh_db=name, time=t, remesh="; ".join(rep))
        mesh = name
        return load_state(st)

    if a.remesh_first:
        saved = remesh_now(saved, "--remesh-first")
    model, prob = build(saved, mesh)
    y = prob.scale(saved.x)
    for c in range(a.chunks):
        yn = prob.phi(y, saved.p, tag=f"adapt{k0 + c + 1}", want_diag=True)
        r = rms(yn - y)
        t += prob.spec.horizon
        y = yn
        f = os.path.join(pdir, f"state_{k0 + c + 1:04d}.npz")
        save_state(f, saved.layout, prob.unscale(y), saved.p, dict(saved.frozen), prob.last_diag, time=t, mesh_db=mesh)
        cur = load_state(f)
        go, note = needs_remesh(cfg, cur, mesh, opt)
        log(f"chunk {k0 + c + 1}  t = {t:g} yr  rms change {r:.3e}  mesh {mesh}; {note}")
        if go:
            saved = remesh_now(cur, "the margin is leaving the fine band")
            model, prob = build(saved, mesh)
            y = prob.scale(saved.x)
        else:
            saved = cur
    log(f"done: t = {t:g} yr, mesh {mesh}")


if __name__ == "__main__":
    main()
