"""
Command line driver. Run inside a batch job (it launches `mpirun ElmerSolver` itself):

    python3 -m ess setup      CONFIG                 mesh (gmsh, ElmerGrid) and initial conditions -> <work>/case
    python3 -m ess init       CONFIG                 first state from those initial conditions -> <work>/state0.npz
    python3 -m ess extract    CONFIG --restart F     first state from an Elmer restart file instead
    python3 -m ess check      CONFIG [--state F]     round trip, repeatability, linearity, residual
    python3 -m ess remask     CONFIG --state F       move the frozen masks (cold/temperate; marine: grounded) to fit a state
    python3 -m ess run        CONFIG [--state F] --chunks N      time integration (masks moved per chunk with [masks] auto)
    python3 -m ess newton     CONFIG [--state F] [--p P]         steady state at fixed parameter
    python3 -m ess transfer   CONFIG --from F [--out G]          carry a state onto this config's mesh
    python3 -m ess continue   CONFIG [--state F] [--name B]      pseudo-arclength branch (resumes)
    python3 -m ess spectrum   CONFIG --state F [--p P]           steady states with temperate beds growing from the terminus
    python3 -m ess stability  CONFIG (--state F | --name B --point K)
    python3 -m ess velocity   CONFIG (--state F [F ...] | --name B [--points K ...])   Stokes velocities
    python3 -m ess toy-demo   [WORKDIR]               whole pipeline on the stand-in model, no Elmer
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from .config import (build_model, continuation_options, fd_settings, load_config, map_spec, newton_options,
                     setup_logging, stability_options)
from .continuation import BranchStore, continuation
from .model import FixedPointProblem, rms
from .newton import newton_krylov
from .stability import stability
from .statefile import SavedState, find_name, load_state, save_state

log = logging.getLogger("ess")


# ---------------------------------------------------------------- helpers

def _saved(cfg, path):
    path = path or os.path.join(cfg["_work"], "state0.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found (run `ess extract` first, or give --state)")
    return load_state(path)


def _toy_saved(model, cfg):
    """Initial state for the toy model: cold-branch steady state at the configured parameter (+ noise)."""
    from .toy import ToyColumn
    assert isinstance(model, ToyColumn)
    p = float(cfg["parameter"]["value"])
    x, _ = model.steady_state(model.steady_Tb(p, cfg["model"].get("start_branch", "cold")))
    x = x + 0.01 * np.sin(np.arange(len(x)))
    return SavedState(model.layout, x, p, {}, {}, {})


def _problem(cfg, model, section="map"):
    eps, scheme = fd_settings(cfg)
    return FixedPointProblem(model, map_spec(cfg, section), fd_eps=eps, fd_scheme=scheme,
                             p_scale=float(cfg["parameter"].get("scale", 1.0)))


def _setup(args, need_state=True):
    cfg = load_config(args.config)
    setup_logging(cfg, getattr(args, "verbose", False))
    log.info(f"config {cfg['_config']}; work dir {cfg['_work']}")
    if cfg["model"].get("type", "elmer") == "toy":
        model = build_model(cfg)
        saved = load_state(args.state) if getattr(args, "state", None) else _toy_saved(model, cfg)
        return cfg, model, saved
    saved = _saved(cfg, getattr(args, "state", None)) if need_state else None
    if saved is not None and saved.frozen:          # frozen fields the config no longer lists: Elmer's StateRead
        keep = {n.lower() for n in cfg["state"].get("frozen", [])}  # stops on fields the model does not have
        drop = [k for k in saved.frozen if k.lower() not in keep]
        if drop:
            log.info(f"state: frozen fields not in [state] frozen, dropped: {', '.join(drop)}")
            saved.frozen = {k: v for k, v in saved.frozen.items() if k.lower() in keep}
    return cfg, build_model(cfg, saved), saved


def _marine(model):
    return bool(getattr(model, "marine", False))


def _mask_cols(cfg, model, saved, x, diag, warn=True):
    if cfg["model"].get("type", "elmer") != "elmer":
        return {}
    from .elmer import mask_report
    rep = mask_report(saved.layout, x, saved.frozen, diag, marine=_marine(model),
                      rho_ratio=getattr(model, "rho_ratio", None))
    if warn and (rep.get("cold_above_Tm") or rep.get("temp_freezing") or rep.get("to_grounded")
                 or rep.get("to_floating")):
        log.warning(f"frozen mask inconsistent here: {rep}")
    return rep


def _masks_auto(cfg, model):
    """[masks] auto (default: on for the marine flowline): after Newton, move the frozen masks to fit
    the converged state and solve again, up to max_rounds times"""
    m = cfg.get("masks", {})
    on = bool(m.get("auto", _marine(model))) and cfg["model"].get("type", "elmer") == "elmer"
    return on, int(m.get("max_rounds", 6))


def _remask_loop(cfg, model, saved, prob, y, p, solve, label):
    """y converged at p: while the masks do not fit, move them (saved.frozen is the model's frozen dict,
    changed in place) and re-solve with solve(y, p) -> result with .y, .converged. Returns (y, result)."""
    from .masks import describe, n_changes, update_masks
    on, rounds = _masks_auto(cfg, model)
    r = None
    if not on:
        return y, r
    for k in range(rounds):
        x = prob.unscale(y)
        xn, frozen, rep = update_masks(saved.layout, x, saved.frozen, prob.last_diag, marine=_marine(model),
                                       rho_ratio=getattr(model, "rho_ratio", None))
        if n_changes(rep) == 0:
            if k:
                log.info(f"{label}: masks consistent after {k} remask round(s)")
            return y, r
        log.info(f"{label}: remask round {k + 1}: {describe(rep)}")
        saved.frozen.clear()
        saved.frozen.update(frozen)
        r = solve(prob.scale(xn), p)
        if not r.converged:
            log.warning(f"{label}: Newton did not converge after remask round {k + 1}")
            return r.y, r
        y = r.y
    log.warning(f"{label}: masks still changing after {rounds} rounds ({describe(rep)})")
    return y, r


# ---------------------------------------------------------------- commands

def cmd_setup(args):
    cfg = load_config(args.config)
    setup_logging(cfg, args.verbose)
    from .case import setup_case
    setup_case(cfg, mesh=not args.ics_only)


def cmd_extract(args):
    cfg = load_config(args.config)
    setup_logging(cfg, args.verbose)
    from .elmer import ElmerModel
    model = ElmerModel(cfg)
    ex = cfg.get("extract", {})
    restart = getattr(args, "restart", None) or (ex.get("restart_file") if args.cmd == "extract" else None)
    if args.cmd == "extract" and not restart:
        raise SystemExit("give the restart file ([extract] restart_file or --restart)")
    rec, layout, frozen = model.extract(restart, bool(ex.get("restart_before_ics", False)))
    model.layout = layout
    x = layout.to_vector(rec)
    if _marine(model):
        # grounded mask at the nodes from the base and the bed (the table interpolation smears the step)
        from .masks import _on
        kg = find_name(frozen, "GroundedMask")
        g_gm = frozen[kg][0]
        zb = _on(*rec[find_name(rec, "Zb")], g_gm)
        bed = _on(*frozen[find_name(frozen, "bedrock")], g_gm)
        gm = np.where(zb - bed < 0.5, 1.0, -1.0)
        frozen[kg] = (g_gm, gm)
        log.info(f"extract: grounded mask from Zb - bed: {int((gm > 0).sum())} grounded, "
                 f"{int((gm < 0).sum())} floating base nodes")
    coords = {k: v for k, v in rec.items() if k.startswith("coordinate")}
    out = args.out or os.path.join(cfg["_work"], "state0.npz")
    p = float(cfg["parameter"]["value"])
    save_state(out, layout, x, p, frozen, coords, source=restart or "initial conditions")
    log.info(f"wrote {out}")
    rep = _mask_cols(cfg, model, load_state(out), x, {})
    if rep:
        log.info(f"masks: {rep}")


def cmd_remask(args):
    """Move the frozen masks to fit a converged state (ess.masks.update_masks): the cold/temperate bed mask
    (cold -> temperate above T0, temperate -> cold where the bed refreezes) and, marine, the grounded mask
    (floating -> grounded where the base reaches the bed, grounded -> floating below flotation)."""
    cfg, model, s = _setup(args)
    from .masks import describe, update_masks
    frozen = dict(s.frozen)
    if args.all_cold:
        k = find_name(frozen, "ColdTempMask")
        g, ct = frozen[k]
        frozen[k] = (g, np.full_like(ct, -1.0))
        log.info(f"remask: all {len(ct)} bed nodes set cold (flux condition), {int((ct > 0).sum())} were temperate")
    if not any(k.lower() == "temperature loads" for k in s.diag):
        log.warning("remask: no Temperature Loads in this state, only cold -> temperate switches")
    x, frozen, rep = update_masks(s.layout, s.x, frozen, s.diag, marine=_marine(model),
                                  rho_ratio=getattr(model, "rho_ratio", None))
    out = args.out or os.path.splitext(args.state or os.path.join(cfg["_work"], "state0.npz"))[0] + "_remask.npz"
    save_state(out, s.layout, x, s.p, frozen, s.diag, **s.meta)
    log.info(f"remask: {describe(rep)}; wrote {out}")


def cmd_check(args):
    cfg, model, saved = _setup(args)
    prob = _problem(cfg, model)
    p = saved.p if args.p is None else args.p
    y = prob.scale(saved.x)
    T = prob.spec.horizon
    is_elmer = cfg["model"].get("type", "elmer") == "elmer"
    if is_elmer and not args.skip_roundtrip:
        from .model import MapSpec
        from .statefile import merge_partitions, partition_files, write_state_file
        import shutil
        run_dir = model._new_run_dir("roundtrip")
        with open(os.path.join(run_dir, "parameters.lua"), "w") as f:
            f.write(model.params_lua(p))
        write_state_file(os.path.join(run_dir, "state_in.bin"), model.state_records(saved.x))
        with open(os.path.join(run_dir, "run.sif"), "w") as f:
            f.write(model.build_sif(MapSpec(prob.spec.dt, 1), state_out0="state0.bin",
                                    write0=model.state_vars + model.frozen_vars, runname="roundtrip"))
        model._launch(run_dir, "roundtrip")
        rec, _ = merge_partitions(partition_files(os.path.join(run_dir, "state0.bin")))
        x0 = saved.layout.to_vector(rec)
        d = np.max(np.abs(x0 - saved.x))
        log.info(f"check roundtrip: max |x_read - x| = {d:.3e}  ({'OK, exact' if d == 0 else 'NOT EXACT'})")
        if not model.keep_runs:
            shutil.rmtree(run_dir, ignore_errors=True)
    cache0 = getattr(model, "use_cache", None)
    if cache0 is not None:
        model.use_cache = False
    p1 = prob.phi(y, p, tag="check_a", want_diag=True)
    p2 = prob.phi(y, p, tag="check_b")
    if cache0 is not None:
        model.use_cache = cache0
    rep = rms(p1 - p2)
    log.info(f"check repeat: rms(Phi_a - Phi_b) = {rep:.3e} (scaled units; FD perturbation {prob.fd_eps:.1e})")
    F = p1 - y
    log.info(f"check residual: rms(Phi(x) - x) = {rms(F):.3e} over T = {T:g}, rate {rms(F) / T:.3e} per unit time")
    for name, sl in saved.layout.slices().items():
        log.info(f"    block {name}: rms {rms(F[sl]):.3e}, max {np.max(np.abs(F[sl])):.3e} (scaled)")
    rng = np.random.default_rng(1)
    if args.vector == "residual":
        v = F.copy()       # the direction Newton/GMRES work in: a kink there shows up here, not in a random v
    elif args.vector == "bed":
        # temperature perturbation at the basal nodes only (where sliding and friction heating switch)
        v = np.zeros(prob.n)
        sl = next(s_ for n_, s_ in saved.layout.slices().items() if n_.lower() == "temperature")
        gT = next(b_.gids for b_ in saved.layout.blocks if b_.name.lower() == "temperature")
        ct = saved.frozen[next(k for k in saved.frozen if k.lower() == "coldtempmask")][0]
        idx = np.searchsorted(gT, ct)
        ok = idx < len(gT)
        idx = idx[ok][gT[idx[ok]] == ct[ok]]
        v[sl][idx] = rng.standard_normal(len(idx))
        log.info(f"check: perturbing the basal temperature at {len(idx)} bed nodes")
    else:
        v = rng.standard_normal(prob.n)
    prev = None
    log.info(f"check linearity ({args.vector} direction): J v for decreasing perturbation size "
             f"(look for a plateau)")
    for e in args.eps:
        prob.fd_eps = e
        jv = prob.dF(y, p, F, v, tag="check_jv")
        msg = f"    eps {e:.0e}: rms(Jv) {rms(jv):.6e}"
        if prev is not None:
            msg += f"   rel. change {rms(jv - prev) / max(rms(jv), 1e-300):.2e}"
        log.info(msg)
        prev = jv
    mc = _mask_cols(cfg, model, saved, saved.x, prob.last_diag)
    if mc:
        log.info(f"masks: {mc}")


def cmd_run(args):
    cfg, model, saved = _setup(args)
    if args.dt is not None:
        cfg.setdefault("map", {})["dt"] = args.dt
    if args.steps is not None:
        cfg.setdefault("map", {})["n_steps"] = args.steps
    cfg.setdefault("map", {})["rock_heat_capacity"] = args.rock_cap
    contact = _marine(model) and args.gl == "contact"
    if contact:
        model.gl_mode = "contact"          # ElmerStreams: the grounding line moves every step
    elif _marine(model):
        model.gl_mode = "frozen"           # --gl frozen also when the config says contact
    prob = _problem(cfg, model)
    log.info(f"run: dt = {prob.spec.dt:g}, {prob.spec.n_steps} steps per chunk, "
             f"rock heat capacity x {prob.spec.rock_heat_capacity:g}"
             + (", grounding line by contact every step (--gl contact)" if contact else ""))
    p = saved.p if args.p is None else args.p
    if args.cartesian_bed:
        model.cartesian_bed = True
        log.info("run: --cartesian-bed: the bed condition in x/y/z components (no normal-tangential velocities)")
    if args.stokes_block_file:
        model.stokes_block_file = args.stokes_block_file
    if args.stokes_preconditioner:
        model.stokes_precond = args.stokes_preconditioner
        log.info(f"run: Stokes preconditioner {args.stokes_preconditioner}")
    if args.stokes_element:
        model.stokes_element = args.stokes_element
        log.info(f"run: Stokes element {args.stokes_element}")
    if args.shelf_cut is not None:
        model.shelf_cut_x = args.shelf_cut * 1e3
        model.shelf_cut_target = not args.shelf_cut_no_target
        log.info(f"run: EXPERIMENTAL passive shelf beyond x = {args.shelf_cut:g} km"
                 + ("" if model.shelf_cut_target else " (no Passive Target boundary: the cut face stress free)"))
    if args.output_level is not None:
        model.output_level = args.output_level
    if args.stokes_solver:
        model.stokes_solver = args.stokes_solver
        log.info(f"run: Stokes module {args.stokes_solver}")
    if args.rotate_guess:
        model.stokes_rotate_guess = True
        log.info("run: the Stokes initial guess rotated to the normal-tangential frame before each solve")
    for nm in args.add_diag or []:
        if nm not in model.diagnostics:
            model.diagnostics.append(nm)
    if args.warm_start:
        ws = load_state(args.warm_start)
        allws = {**ws.diag, **ws.frozen}
        got = []
        for nm in ("Velocity 1", "Velocity 2", "Velocity 3", "Pressure"):
            try:
                k = find_name(allws, nm)
            except KeyError:
                continue
            model.warm_records[nm] = allws[k]
            got.append(nm)
        if not got:
            raise SystemExit(f"--warm-start {args.warm_start}: no Velocity / Pressure in its diagnostics")
        log.info(f"run: warm start from {args.warm_start}: {', '.join(got)}"
                 + ("" if "Pressure" in got else " (no Pressure: write it with --add-diag Pressure)"))
    if args.linear_stokes:
        model.linear_stokes = args.linear_stokes
        log.info(f"run: Stokes linear solver {args.linear_stokes}"
                 + (f" ({model.stokes_block_file})" if args.linear_stokes == "block" else ""))
    if args.friction_heat:
        model.friction_heat = args.friction_heat
        log.info(f"run: friction heating {args.friction_heat}")
    if args.periodic_gl:
        model.periodic_gl = args.periodic_gl
        log.info(f"run: periodic grounding line treatment: {args.periodic_gl}")
    for kv in args.lua or []:               # Lua parameter overrides for this run (e.g. forcing phases)
        k, _, v = kv.partition("=")
        if not k or not v:
            raise SystemExit(f"run: --lua expects NAME=VALUE, not {kv!r}")
        model.lua_overrides[k.strip()] = v.strip()
        log.info(f"run: Lua override {k.strip()} = {v.strip()}")
    if args.stokes_only and args.heat_only:
        raise SystemExit("run: --stokes-only and --heat-only exclude each other")
    if args.heat_only:
        model.heat_only = True
        log.info("run: --heat-only: no free surface (the geometry stays as in the start state); Stokes and heat run")
    if args.stokes_only:
        model.stokes_only = True
        log.info("run: --stokes-only: Stokes and the free surface only (no heat solve, no strain heating); "
                 "the temperature stays as in the start state")
    if args.bed:                            # the whole bed temperate or cold (marine: the shelf base is at T0
        kn = find_name(saved.frozen, "ColdTempMask")      # whatever the mask says: tempbc in ess_utils.lua)
        g_ = saved.frozen[kn][0]
        saved.frozen[kn] = (g_, np.full(len(g_), 1.0 if args.bed == "temperate" else -1.0))
        log.info(f"run: the whole bed {args.bed} (ColdTempMask {'+1' if args.bed == 'temperate' else '-1'} on "
                 f"{len(g_)} nodes); the cold/temperate mask is not moved")
        tb = next(b for b in saved.layout.blocks if b.name.lower() == "temperature")
        j = np.searchsorted(tb.gids, g_)
        ok = (j < len(tb.gids)) & (tb.gids[np.clip(j, 0, len(tb.gids) - 1)] == g_)
        xs = saved.x.copy()
        Tbed = saved.layout.split(xs)[tb.name]               # a view into xs
        if args.bed == "temperate":         # sliding reads the bed temperature: T0 now, not after the first heat solve
            Tbed[j[ok]] = 273.15
            saved.x = xs
            log.info(f"run: bed temperature set to T0 on {int(ok.sum())} bed nodes")
        else:                               # cold: the sliding factor is capped (temperature-independent) only well below T0
            from .masks import _on
            gm = _on(*saved.frozen[find_name(saved.frozen, "GroundedMask")], g_, fill=1.0)
            P_ = {}
            try:
                from .case import eval_lua_parameters
                P_ = eval_lua_parameters(model.params_file)
            except Exception:
                pass
            dTs = float(P_.get("dT", 0.15))
            cap = float(cfg.get("lua", {}).get("ess_slide_maxfactor", 0) or 0)
            Tw = 273.15 - (dTs * np.log(cap) if cap > 0 else 0.0)
            warm = ok & (gm > -0.5) & (Tbed[np.clip(j, 0, len(Tbed) - 1)] > Tw)
            if warm.any() and model.sliding != "temperate":
                log.warning(f"run: {int(warm.sum())} grounded bed nodes are warmer than {Tw:.2f} K, where the sliding "
                            f"factor still depends on temperature" + (" (--stokes-only keeps their temperature fixed)"
                                                                       if args.stokes_only else ""))
    y = prob.scale(saved.x)
    on, _ = _masks_auto(cfg, model)
    remask = on if args.remask is None else args.remask
    if args.bed and remask:
        log.info("run: --bed: no remask of the cold/temperate mask (the grounding line still moves with --gl contact)")
        remask = False
    if remask:
        log.info("run: the frozen masks are moved after every chunk (cold/temperate"
                 + (", grounding line)" if _marine(model) else ")"))
    from .masks import describe, n_changes, update_masks
    out = args.out or os.path.join(cfg["_work"], f"run_p{p:g}.npz")
    sdir = os.path.join(os.path.splitext(out)[0] + "_chunks", "points")
    if args.save_every:
        os.makedirs(sdir, exist_ok=True)
        log.info(f"run: states every {args.save_every} chunk(s) to {sdir}/state_XXXX.npz "
                 f"(plots/plot_branch_states.py with BRANCH = {os.path.dirname(sdir)})")
    t_end = 0.0
    for c in range(args.chunks):
        yn = prob.phi(y, p, tag=f"run{c}", want_diag=True)
        t = (c + 1) * prob.spec.horizon
        dy = rms(yn - y)
        log.info(f"run: chunk {c + 1}/{args.chunks}  t = {t:g}  rms change {dy:.3e}")
        y = yn
        if contact:                         # the grounded mask Elmer ended the chunk with
            from .masks import _on
            k = find_name(saved.frozen, "GroundedMask")
            g = saved.frozen[k][0]
            gm_new = _on(*prob.last_diag[find_name(prob.last_diag, "GroundedMask")], g, fill=-1.0)
            gm_old = saved.frozen[k][1]
            saved.frozen[k] = (g, gm_new)
            xg = _on(*prob.last_diag[find_name(prob.last_diag, "coordinate 1")], g)
            if np.any((gm_new > -0.5) != (gm_old > -0.5)):
                log.info(f"run: contact grounding line {xg[gm_old > -0.5].max() / 1e3:.2f} -> "
                         f"{xg[gm_new > -0.5].max() / 1e3:.2f} km")
        if getattr(model, "thermal", "frozen") == "elmerstreams":
            # ElmerStreams thermal chain: Elmer moves the cold/temperate mask and the water sheet every step;
            # carry what it ended the chunk with into the next chunk (as the grounded mask above)
            from .masks import _on
            for nm in ("ColdTempMask", "Water Sheet Thickness"):
                try:
                    kf, kd = find_name(saved.frozen, nm), find_name(prob.last_diag, nm)
                except KeyError:
                    log.warning(f"run: thermal = elmerstreams: {nm} not in the frozen fields / diagnostics: not carried")
                    continue
                g, old = saved.frozen[kf]
                new = _on(*prob.last_diag[kd], g, fill=np.nan)
                miss = np.isnan(new)
                if miss.any():                  # nodes Elmer did not write: keep their old values
                    log.warning(f"run: {nm}: {int(miss.sum())} nodes without an Elmer value, old values kept")
                    new[miss] = old[miss]
                saved.frozen[kf] = (g, new)
        # the state after this chunk, with the masks it ran with (before the remask)
        save_state(out, saved.layout, prob.unscale(y), p, dict(saved.frozen), prob.last_diag, time=t)
        if args.save_every and (c + 1) % args.save_every == 0:
            save_state(os.path.join(sdir, f"state_{c + 1 + args.chunk_offset:04d}.npz"), saved.layout, prob.unscale(y), p,
                       dict(saved.frozen), prob.last_diag, time=t)
        t_end = t
        if args.until_tol and dy < args.until_tol:
            log.info(f"run: rms change {dy:.3e} < {args.until_tol:g} over one chunk ({prob.spec.horizon:g} yr): "
                     f"steady enough, stopping at t = {t:g}")
            break
        if remask:
            xn, frozen, rep = update_masks(saved.layout, prob.unscale(y), saved.frozen, prob.last_diag,
                                           marine=_marine(model), rho_ratio=getattr(model, "rho_ratio", None),
                                           grounding=not contact,
                                           periodic_width=(model.width if getattr(model, "periodic_sides", False)
                                                           else None))
            if n_changes(rep):
                log.info(f"run: remask after chunk {c + 1}: {describe(rep) if not contact else describe({k_: v_ for k_, v_ in rep.items() if not k_.startswith(('x_gl', 'to_gr', 'to_fl'))})}")
                saved.frozen.clear()
                saved.frozen.update(frozen)
                y = prob.scale(xn)
    nsteps_done = int(round(t_end / prob.spec.dt)) if prob.spec.dt else 0
    save_state(out, saved.layout, prob.unscale(y), p, saved.frozen, prob.last_diag, time=t_end,
               elmer_seconds=prob.t_model, elmer_runs=prob.nfev, steps=nsteps_done,
               linear_stokes=str(getattr(model, "linear_stokes", "")))
    log.info(f"wrote {out}; Elmer {prob.t_model:.0f} s for {prob.nfev} runs, {nsteps_done} steps"
             + (f" ({prob.t_model / nsteps_done:.2f} s per step)" if nsteps_done else ""))


def cmd_newton(args):
    cfg, model, saved = _setup(args)
    prob = _problem(cfg, model)
    p = saved.p if args.p is None else args.p
    ck = os.path.join(cfg["_work"], "newton_checkpoint.npz")
    r = newton_krylov(prob, prob.scale(saved.x), p, newton_options(cfg), checkpoint=ck)
    if r.converged:
        def solve(y, p):
            return newton_krylov(prob, y, p, newton_options(cfg), checkpoint=ck, label="newton (remasked)")
        _, r2 = _remask_loop(cfg, model, saved, prob, r.y, p, solve, "newton")
        r = r2 or r
    x = prob.unscale(r.y)
    out = args.out or os.path.join(cfg["_work"], f"steady_p{p:g}.npz")
    mc = _mask_cols(cfg, model, saved, x, prob.last_diag)
    save_state(out, saved.layout, x, p, saved.frozen, prob.last_diag, converged=r.converged,
               rms_F=r.history[-1], newton_its=r.iterations)
    log.info(f"newton: {'converged' if r.converged else 'NOT converged'} in {r.iterations} its, "
             f"{prob.nfev} model runs ({prob.t_model / 3600:.2f} h); wrote {out}" + (f"; masks {mc}" if mc else ""))
    return 0 if r.converged else 1


def _hydro_params(cfg):
    hy = cfg.get("hydrology", {})
    return dict(K0=float(hy.get("K0", 1.0)), alpha=float(hy.get("alpha", 3.0)), h0=float(hy.get("h0", 0.1)),
                c_N=float(hy.get("c_N", 1.0)), h_max=float(hy.get("h_max", 1e3)))


def _patch_diagnostics(model):
    """the bed pressure (phi0) and the melt rate (hydrology on a temperate patch) need these"""
    model.diagnostics = list(dict.fromkeys(model.diagnostics + ["Pressure", "Temperature Loads",
                                                                "Temperature Boundary Weights"]
                                           + (["Strain Heating Boundary Weights"] if model.strain_heat else [])))


def cmd_continue(args):
    cfg, model, saved = _setup(args)
    prob = _problem(cfg, model)
    cc = cfg.get("continuation", {})
    name = args.name or cc.get("name", "main")
    bdir = os.path.join(cfg["_work"], f"branch_{name}")
    # --temperate-k K: a frozen temperate patch of K bed columns from the terminus of the START state (all
    # other bed nodes cold), kept fixed along the branch; every point gets the patch hydrology and the checks
    patch = None
    tk = getattr(args, "temperate_k", None)
    if tk is not None:
        if cfg["model"].get("type", "elmer") != "elmer":
            raise SystemExit("--temperate-k needs the Elmer model")
        if _masks_auto(cfg, model)[0]:
            raise SystemExit("--temperate-k: set [masks] auto = false (the patch must stay frozen)")
        from .case import eval_lua_parameters
        from .spectrum import bed_columns, mask_for
        g_ct, col, ux = bed_columns(saved.layout, saved.frozen, saved.diag, x=saved.x,
                                    edge_thickness=args.edge_thickness)
        if tk > len(ux):
            raise SystemExit(f"--temperate-k {tk}: only {len(ux)} glaciated bed columns")
        saved.frozen[find_name(saved.frozen, "ColdTempMask")] = (g_ct, mask_for(tk, g_ct, col))
        _patch_diagnostics(model)
        PP = eval_lua_parameters(model.params_file)
        patch = dict(g_ct=g_ct, col=col, ux=ux, k=tk, hydro=_hydro_params(cfg),
                     phys=dict(rhoi=PP["rhoi"], Lw=PP["Lw"], rhow=PP["rhow"], gravity=PP["gravity"]))
        log.info(f"continue: temperate patch of {tk} bed column(s) from the terminus at x = {ux[0] / 1e3:.2f} km "
                 + (f"(ice thicker than {args.edge_thickness:g} m)" if args.edge_thickness is not None
                    else "(the frozen GroundedMask)") + f", cold/temperate boundary at {ux[tk - 1] / 1e3:.2f} km"
                 if tk > 0 else "continue: --temperate-k 0: all-cold bed")
    every = int(cc.get("stability_every", 0))
    sopt = stability_options(cfg)
    sspec = map_spec(cfg, "stability_map")

    def on_save(k, y, p, row):
        x = prob.unscale(y)
        cols = _mask_cols(cfg, model, saved, x, prob.last_diag)
        hyd = {}
        if patch is not None:
            from .spectrum import patch_hydrology
            ok, h_b, mm, note, ex, hyd = (patch_hydrology(prob.last_diag, saved.layout, x, patch["g_ct"],
                                                          patch["col"], patch["ux"], patch["k"],
                                                          hydro=patch["hydro"], **patch["phys"])
                                          if patch["k"] > 0 else (True, np.nan, np.nan, "", {}, {}))
            cols.update(ex)
            from .spectrum import basal_temperature
            Tb = basal_temperature(saved.layout, x, patch["g_ct"])
            cold = (patch["col"] >= patch["k"])                 # glaciated columns outside the patch
            cols["cold_above_Tm"] = int((cold & (Tb > 273.15)).sum())
            cols["Tb_cold_max"] = float(np.nanmax(Tb[cold])) if cold.any() else np.nan
            cols.update(temperate_k=patch["k"], hydro_ok=int(ok), h_boundary=h_b, melt_mean=mm,
                        x_boundary_km=float(patch["ux"][patch["k"] - 1] / 1e3) if patch["k"] else np.nan,
                        thermal_ok=int(not cols.get("cold_above_Tm")), note=note)
            log.info(f"continue: point {k} p = {p:.6g}: cold nodes above T0 {cols.get('cold_above_Tm', 0)}, "
                     f"hydrology {'ok' if ok else 'FAILS'}, h at the boundary {h_b:.3g} m" + (f" ({note})" if note else ""))
        extra = {}
        if every and k % every == 0:
            sopt_small = type(sopt)(**{**sopt.__dict__, "m": min(sopt.m, int(cc.get("stability_m", 12)))})
            st = stability(prob, y, p, sspec, sopt_small)
            extra = dict(n_unstable=st.n_unstable, sigma_max=float(np.max(st.sigma.real)),
                         mu_max=float(np.max(np.abs(st.mu))))
            cols.update(extra)
        save_state(os.path.join(bdir, "points", f"state_{k:04d}.npz"), saved.layout, x, p, saved.frozen,
                   {**prob.last_diag, **hyd}, s=row["s"],
                   **{c: v for c, v in cols.items() if np.isscalar(v) and not isinstance(v, str)})
        return cols

    store = BranchStore(bdir, prob, on_save=on_save)
    if len(store):
        log.info(f"continue: resuming {bdir} at point {len(store)}")
    copt = continuation_options(cfg)
    ds_override = getattr(args, "ds", None)
    if ds_override is not None:
        copt.ds = ds_override
        if store.rows:
            store.rows[-1]["ds"] = ds_override
        log.info(f"continue: starting with ds = {ds_override:g}")
    on, _ = _masks_auto(cfg, model)
    refine = None
    if on:
        from dataclasses import replace as _replace
        nopt = _replace(newton_options(cfg), tol=copt.tol)

        def refine(y, p):
            """after each converged point: move the masks and re-converge at the same p"""
            return _remask_loop(cfg, model, saved, prob, y, p,
                                lambda yy, pp: newton_krylov(prob, yy, pp, nopt, label=f"remask p={pp:.6g}"),
                                f"continue p={p:.6g}")
        log.info("continue: [masks] auto: the frozen masks are moved after each point (GL / cold-temperate)")
    continuation(prob, prob.scale(saved.x), saved.p if args.p is None else args.p, store,
                 copt, newton_options(cfg), refine=refine)
    log.info(f"continue: {len(store)} points in {bdir}/branch.csv; {prob.nfev} model runs "
             f"({prob.t_model / 3600:.2f} h)")


def cmd_spectrum(args):
    """Steady states at one parameter value with temperate beds of 0, 1, 2, ... columns from the terminus
    (ess/spectrum.py): inconsistent ones (cold bed above melting; water sheet < 0 at the cold/temperate
    boundary) are discarded. With --p-step: the same at p, p + dp, ... (to --p-end), each starting from the
    previous parameter's all-cold steady state; --stop-after-dry N moves on after N hydrology failures."""
    cfg, model, saved = _setup(args)
    if cfg["model"].get("type", "elmer") != "elmer":
        raise SystemExit("spectrum needs the Elmer model")
    from dataclasses import replace as _replace
    from .case import eval_lua_parameters
    from .spectrum import run_spectrum
    from .statefile import load_state
    prob = _problem(cfg, model)
    p0 = saved.p if args.p is None else args.p
    P = eval_lua_parameters(model.params_file)
    nopt = newton_options(cfg)
    if args.maxit:
        nopt = _replace(nopt, maxit=args.maxit)
    hydro = _hydro_params(cfg)
    log.info(f"spectrum: hydrology K(h) = {hydro['K0']:g} h^{hydro['alpha']:g}, h0 = {hydro['h0']:g} m at the "
             f"margin, potential {hydro['c_N']:g}/h - phi0")
    _patch_diagnostics(model)
    if args.p_step:
        p_end = args.p_end if args.p_end is not None else float(cfg.get("continuation", {}).get("p_max", p0))
        n = int(np.floor((p_end - p0) / args.p_step + 1e-9)) + 1
        plist = [p0 + i * args.p_step for i in range(max(n, 1))]
        base = args.out or cfg["_work"]
        tag = f"_e{args.edge_thickness:g}" if args.edge_thickness is not None else ""
        outs = [os.path.join(base, f"spectrum_p{p:g}{tag}") for p in plist]
        log.info(f"spectrum sweep: p = {', '.join(f'{p:g}' for p in plist)}")
    else:
        plist = [p0]
        outs = [args.out or os.path.join(cfg["_work"], f"spectrum_p{p0:g}")]
    y0 = prob.scale(saved.x)
    for p, out in zip(plist, outs):
        rows = run_spectrum(prob, saved, y0, p, nopt, out, k_max=args.k_max, k_step=args.k_step,
                            tol_T=args.tol_T, rhoi=P["rhoi"], Lw=P["Lw"], rhow=P["rhow"], gravity=P["gravity"],
                            hydro=hydro, edge_thickness=args.edge_thickness, stop_after_dry=args.stop_after_dry)
        kept = [r for r in rows if str(r["kept"]) == "1"]
        log.info(f"spectrum: p = {p:g}: {len(kept)} consistent states of {len(rows)} tried; {out}/spectrum.csv")
        f0 = os.path.join(out, "k0_state.npz")
        if not os.path.exists(f0):
            log.warning(f"spectrum: no converged all-cold state at p = {p:g}; stopping the sweep")
            break
        s0 = load_state(f0)                      # the next parameter starts from this all-cold steady state
        y0 = prob.scale(s0.x)
        saved.diag = s0.diag


def _spectrum_ps(work):
    """the Tsurf values of every spectrum folder under work (spectrum_p<T>..., also in subfolders)"""
    import glob
    import re
    ps = set()
    for f in glob.glob(os.path.join(work, "**", "spectrum_p*"), recursive=True):
        m = re.match(r"spectrum_p([0-9.]+)", os.path.basename(f))
        if m and os.path.isdir(f):
            ps.add(round(float(m.group(1)), 6))
    return sorted(ps)


def cmd_temperate_sweep(args):
    """The fully temperate bed (T = T0 on the whole bed, beyond the margin too): with a constant viscosity
    and sliding that depends on the basal temperature only, the flow and the geometry no longer depend on
    Tsurf. (1) converge that state at the start Tsurf: transient chunks (until --spinup-tol), then Newton;
    (2) Tsurf sweep: Newton
    at each Tsurf from the nearest finished one (only the temperature should change), the patch hydrology
    on the glaciated bed (from the terminus: ice thicker than --edge-thickness), and the geometry change from the start state as a check of the decoupling.
    Written as a branch (<work>/branch_<name>/branch.csv, points/state_XXXX.npz) for plots/plot_spectrum.py."""
    if not args.state:                      # default: this sweep's own converged start state
        st0 = os.path.join(load_config(args.config)["_work"], f"branch_{args.name}", "temperate_steady.npz")
        if not os.path.exists(st0):
            raise SystemExit(f"temperate-sweep: no {st0} yet: give --state (a state to spin up from)")
        args.state = st0
    cfg, model, saved = _setup(args)
    if cfg["model"].get("type", "elmer") != "elmer":
        raise SystemExit("temperate-sweep needs the Elmer model")
    if _masks_auto(cfg, model)[0]:
        raise SystemExit("temperate-sweep: set [masks] auto = false (the temperate mask is set here)")
    from dataclasses import replace as _replace
    from .case import eval_lua_parameters
    from .spectrum import bed_columns, mask_for, patch_hydrology
    if args.dt is not None:
        cfg.setdefault("map", {})["dt"] = args.dt
    if args.steps is not None:
        cfg.setdefault("map", {})["n_steps"] = args.steps
    _patch_diagnostics(model)
    prob = _problem(cfg, model)
    PP = eval_lua_parameters(model.params_file)
    phys = dict(rhoi=PP["rhoi"], Lw=PP["Lw"], rhow=PP["rhow"], gravity=PP["gravity"])
    hydro = _hydro_params(cfg)
    nopt = newton_options(cfg)
    if args.maxit:
        nopt = _replace(nopt, maxit=args.maxit)
    kname = find_name(saved.frozen, "ColdTempMask")
    et = args.edge_thickness
    bdir = os.path.join(cfg["_work"], f"branch_{args.name}")
    os.makedirs(os.path.join(bdir, "points"), exist_ok=True)
    p0 = saved.p if args.p is None else args.p

    g_all = saved.frozen[kname][0]
    saved.frozen[kname] = (g_all, np.ones(len(g_all)))      # the WHOLE bed temperate, beyond the margin too

    def fit_mask(x, diag):
        """the glaciated columns (terminus of x: ice thicker than --edge-thickness), for the hydrology and
        the reports; the mask itself is temperate everywhere. Returns (columns, terminus moved)"""
        g, col, ux = bed_columns(saved.layout, saved.frozen, diag, x=x, edge_thickness=et)
        moved = fit_mask.n is not None and fit_mask.n != len(ux)
        fit_mask.n = len(ux)
        return (g, col, ux), moved
    fit_mask.n = None

    # ---- (1) the fully temperate steady state at p0
    start = os.path.join(bdir, "temperate_steady.npz")
    if os.path.exists(start) and not args.restart:
        s0 = load_state(start)
        saved.frozen.clear()
        saved.frozen.update(s0.frozen)
        y, p0, diag0 = prob.scale(s0.x), s0.p, s0.diag
        saved.frozen[kname] = (g_all, np.ones(len(g_all)))
        cols, _ = fit_mask(s0.x, s0.diag)
        log.info(f"temperate-sweep: start from {start} (p = {p0:g}); --restart to converge it again")
    else:
        y, diag = prob.scale(saved.x), saved.diag
        cols, _ = fit_mask(saved.x, diag)
        log.info(f"temperate-sweep: p = {p0:g}, the whole bed temperate ({len(g_all)} bed nodes); "
                 f"{len(cols[2])} glaciated columns (terminus {cols[2][0] / 1e3:.2f} km)")
        nmax = args.chunks or (1000 if args.spinup_tol else 0)
        if args.spinup_tol:
            log.info(f"temperate-sweep: spin-up until the rms change per chunk ({prob.spec.horizon:g} yr) "
                     f"< {args.spinup_tol:g} (at most {nmax} chunks)")
        for c in range(nmax):
            yn = prob.phi(y, p0, tag=f"spin{c}", want_diag=True)
            dy = rms(yn - y)
            log.info(f"temperate-sweep: spin-up chunk {c + 1}/{nmax}, rms change {dy:.3e}")
            y = yn
            cols, ch = fit_mask(prob.unscale(y), prob.last_diag)
            if ch:
                log.info(f"temperate-sweep: terminus now {cols[2][0] / 1e3:.2f} km ({len(cols[2])} glaciated columns)")
            t_spin = float(getattr(saved, "meta", {}).get("spinup_time", 0.0)) + (c + 1) * prob.spec.horizon
            save_state(os.path.join(bdir, "spinup.npz"), saved.layout, prob.unscale(y), p0, dict(saved.frozen),
                       prob.last_diag, spinup_time=t_spin)
            log.info(f"temperate-sweep: spin-up time {t_spin:g} yr (wrote {bdir}/spinup.npz)")
            if args.spinup_tol and dy < args.spinup_tol:
                log.info(f"temperate-sweep: rms change {dy:.3e} < {args.spinup_tol:g}: spin-up done")
                break
        else:
            if args.spinup_tol and nmax:
                log.warning(f"temperate-sweep: spin-up tolerance not reached in {nmax} chunks; Newton anyway")
        if args.spinup_only:
            log.info(f"temperate-sweep: --spinup-only: stopping; continue with --state {bdir}/spinup.npz")
            return
        r = newton_krylov(prob, y, p0, nopt, label="temperate newton")
        if not r.converged:
            raise SystemExit(f"temperate-sweep: Newton did not converge at p = {p0:g} (spin up further: a "
                             "smaller --spinup-tol)")
        y = r.y
        cols, _ = fit_mask(prob.unscale(y), prob.last_diag)
        log.info(f"temperate-sweep: steady at p = {p0:g}: terminus {cols[2][0] / 1e3:.2f} km "
                 f"({len(cols[2])} glaciated columns)")
        diag0 = prob.last_diag
        save_state(start, saved.layout, prob.unscale(y), p0, dict(saved.frozen), diag0)
        log.info(f"temperate-sweep: wrote {start}")
    names = {b.name.lower(): b.name for b in saved.layout.blocks}
    zsn = names.get("zs")
    Zs0 = saved.layout.split(prob.unscale(y))[zsn].copy() if zsn else None

    # ---- (2) the Tsurf sweep
    if args.p_list:
        plist = sorted(float(v) for v in args.p_list.split(","))
    elif args.p_step:
        a, b = args.p_start if args.p_start is not None else p0, args.p_end if args.p_end is not None else p0
        n = int(np.floor((b - a) / args.p_step + 1e-9)) + 1
        plist = [a + i * args.p_step for i in range(max(n, 1))]
    else:
        plist = _spectrum_ps(cfg["_work"])
        log.info(f"temperate-sweep: the Tsurf of the spectrum folders: {', '.join(f'{v:g}' for v in plist)}")
    if not plist:
        raise SystemExit("temperate-sweep: no Tsurf values (--p-list, --p-step, or spectrum folders)")
    csvf = os.path.join(bdir, "branch.csv")
    import csv
    rows = []
    if os.path.exists(csvf):
        with open(csvf) as fh:
            rows = list(csv.DictReader(fh))
    done = {round(float(r["p"]), 6): int(r["index"]) for r in rows}
    states = {p0: y}
    for r_ in rows:                                         # resume: the finished states as starting points
        sf = os.path.join(bdir, "points", f"state_{int(r_['index']):04d}.npz")
        if str(r_.get("converged")) == "1" and os.path.exists(sf):
            states[float(r_["p"])] = prob.scale(load_state(sf).x)
    # --heat-only: the steady temperature on the frozen geometry by one Elmer run of a few very long
    # implicit steps without the free surface (BDF1, linear heat equation: a step of dt -> infinity is the
    # steady solve), then ONE ordinary map evaluation as the check: accepted when its residual is below the
    # Newton tolerance (its diagnostics, e.g. the melt rate, are then those of the full map, as in Newton);
    # otherwise Newton from there
    from .model import MapSpec
    hspec = MapSpec(float(args.heat_dt), int(args.heat_steps), 1, prob.spec.rock_heat_capacity)
    r_pred = None
    if args.heat_only:
        if _marine(model):
            raise SystemExit("--heat-only: land flowline only (the marine template has its own free surfaces)")
        log.info(f"temperate-sweep: --heat-only: {hspec.n_steps} step(s) of {hspec.dt:g} yr without the free "
                 f"surface per Tsurf, then one map evaluation as the check (Newton tolerance {nopt.tol:g})")

    class _Res:
        def __init__(self, y, F, its):
            self.y, self.converged, self.iterations = y, True, its
            self.history, self.F, self.gmres_iters = [rms(F)], F, []

    def heat_only_solve(y_start, p):
        nonlocal r_pred
        model.heat_only = True
        try:
            yh = prob.phi(y_start, p, tag=f"heat p={p:g}", spec=hspec)
        finally:
            model.heat_only = False
        if zsn:
            dz = np.max(np.abs(saved.layout.split(prob.unscale(yh))[zsn] - saved.layout.split(prob.unscale(y_start))[zsn]))
            if dz > 1e-6:
                log.warning(f"temperate-sweep: heat-only run changed Zs by {dz:.3g} m (free surface not off?)")
        F = prob.F(yh, p, tag=f"check p={p:g}", want_diag=True)
        res = rms(F)
        r_pred = yh
        if res <= nopt.tol:
            log.info(f"temperate-sweep: p = {p:g}: heat-only state is steady (residual {res:.2e} <= {nopt.tol:g})")
            return _Res(yh, F, 0)
        log.info(f"temperate-sweep: p = {p:g}: heat-only residual {res:.2e} > {nopt.tol:g}: Newton from there")
        return None

    # from p0 upwards, then from p0 downwards: each start is the neighbouring finished Tsurf
    order = sorted(v for v in plist if v >= p0) + sorted((v for v in plist if v < p0), reverse=True)
    COLS = ["index", "p", "converged", "newton_its", "rms_F", "temperate_k", "n_columns", "x_terminus_km",
            "cold_above_Tm", "temp_freezing", "hydro_ok", "hydro_status", "h_boundary", "h_min", "q_margin",
            "melt_mean", "n_melt_filled", "dZs_max", "dZs_rms", "note"]
    for p in order:
        if round(p, 6) in done:
            continue
        pn = min(states, key=lambda q: abs(q - p))
        log.info(f"temperate-sweep: p = {p:g} from the state at {pn:g}")
        r = None
        if args.heat_only:
            r = heat_only_solve(states[pn], p)
        if r is None:
            r = newton_krylov(prob, states[pn] if not args.heat_only else r_pred, p, nopt, label=f"temperate p={p:g}")
        x = prob.unscale(r.y)
        diag = prob.last_diag
        g, col, ux = bed_columns(saved.layout, saved.frozen, diag, x=x, edge_thickness=et)
        ncol = len(ux)
        row = dict(index=len(rows), p=p, converged=int(r.converged), newton_its=r.iterations,
                   rms_F=r.history[-1] if r.history else np.nan, temperate_k=ncol, n_columns=ncol,
                   x_terminus_km=float(ux[0] / 1e3), cold_above_Tm=0)
        hyd = {}
        if r.converged:
            states[p] = r.y
            # refreezing temperate nodes (heat removed to hold T0): information only; on the temperate bed
            # freezing is a matter for the hydrology (a negative melt rate), not for the mask
            mr = _mask_cols(cfg, model, saved, x, diag, warn=False)
            row["temp_freezing"] = mr.get("temp_freezing", "")
            ok, h_b, mm, note, ex, hyd = patch_hydrology(diag, saved.layout, x, g, col, ux, ncol,
                                                         hydro=hydro, **phys)
            row.update(ex)
            row.update(hydro_ok=int(ok), h_boundary=h_b, melt_mean=mm, note=note)
            if Zs0 is not None:
                dz = saved.layout.split(x)[zsn] - Zs0
                row.update(dZs_max=float(np.max(np.abs(dz))), dZs_rms=float(np.sqrt(np.mean(dz ** 2))))
            if ncol != len(cols[2]):
                row["note"] = (note + "; " if note else "") + f"terminus moved: {ncol} glaciated columns ({len(cols[2])} at p0)"
            save_state(os.path.join(bdir, "points", f"state_{row['index']:04d}.npz"), saved.layout, x, p,
                       dict(saved.frozen), {**diag, **hyd}, temperate_k=ncol)
            log.info(f"temperate-sweep: p = {p:g}: converged in {r.iterations} its; hydrology "
                     f"{'ok' if ok else 'FAILS'} (h at the divide end {h_b:.3g} m{', ' + note if note else ''}); "
                     f"geometry change from p = {p0:g}: max |dZs| {row.get('dZs_max', np.nan):.3g} m")
        else:
            row["note"] = "Newton did not converge"
            log.warning(f"temperate-sweep: p = {p:g}: Newton did not converge")
        rows.append({c: row.get(c, "") for c in COLS})
        with open(csvf + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(csvf + ".tmp", csvf)
    log.info(f"temperate-sweep: {sum(str(r_['converged']) == '1' for r_ in rows)} of {len(rows)} converged; {csvf}")


def cmd_spectrum_down(args):
    """From a (fully) temperate steady state (e.g. a temperate-sweep point at 273.15 K): temperate patches of
    n, n - k_step, ... columns from the terminus, the cold bed spreading from the divide; full Newton for each
    (Stokes and heat coupled), the step cut down to 1 where Newton fails. A spectrum folder for the plots."""
    cfg, model, saved = _setup(args)
    if cfg["model"].get("type", "elmer") != "elmer":
        raise SystemExit("spectrum-down needs the Elmer model")
    if _masks_auto(cfg, model)[0]:
        raise SystemExit("spectrum-down: set [masks] auto = false (the patches are set here)")
    from dataclasses import replace as _replace
    from .case import eval_lua_parameters
    from .spectrum import run_spectrum_down
    _patch_diagnostics(model)
    prob = _problem(cfg, model)
    p = saved.p if args.p is None else args.p
    P = eval_lua_parameters(model.params_file)
    nopt = newton_options(cfg)
    if args.maxit:
        nopt = _replace(nopt, maxit=args.maxit)
    tag = f"_e{args.edge_thickness:g}" if args.edge_thickness is not None else ""
    out = args.out or os.path.join(cfg["_work"], f"spectrum_p{p:g}{tag}_down")
    rows = run_spectrum_down(prob, saved, prob.scale(saved.x), p, nopt, out, k_step=args.k_step, k_min=args.k_min,
                             tol_T=args.tol_T, rhoi=P["rhoi"], Lw=P["Lw"], rhow=P["rhow"], gravity=P["gravity"],
                             hydro=_hydro_params(cfg), edge_thickness=args.edge_thickness)
    kept = [r for r in rows if str(r["kept"]) == "1"]
    log.info(f"spectrum-down: {len(kept)} consistent states of {len(rows)}; {out}/spectrum.csv")


def cmd_transfer(args):
    """Carry a converged state (any config, e.g. the regularised full domain) onto this config's mesh."""
    cfg = load_config(args.config)
    setup_logging(cfg, args.verbose)
    from .transfer import transfer
    src = load_state(args.source)
    dst = _saved(cfg, args.state)
    c = cfg.get("case", {})
    x_end = float(c.get("x_end", 1.0))
    x_end_m = None
    if x_end < 1.0:
        from .case import eval_lua_parameters
        P = eval_lua_parameters(cfg["_path"](c.get("parameters", "case/parameters.lua")))
        x_end_m = x_end * P["L"]
    x, frozen = transfer(src, dst, x_end_m, float(c.get("edge_thickness", 30.0)))
    out = args.out or os.path.join(cfg["_work"], f"start_p{src.p:g}.npz")
    save_state(out, dst.layout, x, src.p, frozen, dst.diag, source=args.source)
    log.info(f"wrote {out} (p = {src.p:g}); next: ess newton {args.config} --state {out}")


def cmd_stability(args):
    cfg = load_config(args.config)
    if args.name is not None:
        args.state = os.path.join(cfg["_work"], f"branch_{args.name}", "points", f"state_{args.point:04d}.npz")
    cfg, model, saved = _setup(args)
    prob = _problem(cfg, model)
    spec = map_spec(cfg, args.map)
    opt = stability_options(cfg)
    if args.m:
        opt.m = args.m
    label = os.path.splitext(os.path.basename(args.state or "state0"))[0]
    if args.name is not None:
        label = f"branch_{args.name}_{label}"
    if args.map != "stability_map":
        label += f"_{args.map}"
    sdir = os.path.join(cfg["_work"], "stability", label)
    os.makedirs(sdir, exist_ok=True)
    y = prob.scale(saved.x)
    span = None
    if args.spanwise:
        from .spanwise import Spanwise
        span = Spanwise(saved.layout, saved.diag)
        sdir += "_spanwise"
        os.makedirs(sdir, exist_ok=True)
        log.info(f"stability: spanwise modes on a strip of width {span.W / 1e3:g} km, {span.ny} nodes across "
                 f"(n = 1..{span.ny - 1}, wavelengths {2 * span.W / 1e3:g}..{2 * span.W / (span.ny - 1) / 1e3:.3g} km); "
                 f"the across-flow mean (n = 0, the 2D modes) is projected out")
    st = stability(prob, y, saved.p, spec, opt, checkpoint=os.path.join(sdir, "arnoldi.npz"),
                   project=span.project if span else None)
    vec = st.vectors * prob.s[:, None]
    extra = {}
    if span:
        spec_n = np.array([span.spectrum(st.vectors[:, i]) for i in range(st.vectors.shape[1])])
        n_dom = spec_n[:, 1:].argmax(axis=1) + 1
        k = n_dom * np.pi / span.W
        lam = 2 * span.W / n_dom
        for i in range(len(st.mu)):
            log.info(f"  mode {i}: |mu| {abs(st.mu[i]):.6f}  sigma {st.sigma[i].real:+.4e}/yr  dominant n = {n_dom[i]} "
                     f"(k_y {k[i] * 1e3:.4g}/km, wavelength {lam[i] / 1e3:.4g} km, {spec_n[i, n_dom[i]]:.0%} of the energy)"
                     f"{'' if st.converged[i] else '  (not converged)'}")
        extra = dict(spanwise_spectrum=spec_n, spanwise_n=n_dom, spanwise_k=k, spanwise_wavelength=lam,
                     spanwise_width=span.W)
    out = os.path.join(sdir, "stability.npz")
    d = saved.layout.to_npz_dict()
    d.update(mu=st.mu, sigma=st.sigma, sigma_be=st.sigma_be, residual=st.residual, converged=st.converged,
             vectors=vec, x=saved.x, p=saved.p, dt=spec.dt, n_steps=spec.n_steps, drift=st.drift, **extra)
    for k_, (g, v) in saved.frozen.items():          # bed nodes etc. for the plots
        d[f"frozen_gids::{k_}"] = g
        d[f"frozen_vals::{k_}"] = v
    for k_, (g, v) in saved.diag.items():
        if k_.lower().startswith("coordinate") and f"diag_gids::{k_}" not in d:
            d[f"diag_gids::{k_}"] = g
            d[f"diag_vals::{k_}"] = v
    for k, (g, v) in prob.last_diag.items():
        d[f"diag_gids::{k}"] = g
        d[f"diag_vals::{k}"] = v
    np.savez(out, **d)
    log.info(f"stability: {st.n_unstable} unstable converged mode(s); wrote {out}")


def cmd_velocity(args):
    """Solve Stokes on saved states and write <state>_vel.npz: the state with Velocity 1-3 (and the
    usual diagnostics) evaluated on exactly that state, for plotting. One time step of 1e-12 yr: the
    velocity comes from the Stokes solve at the start of the step, so T and Zs are those of the file."""
    import glob
    from .model import MapSpec
    cfg = load_config(args.config)
    if args.name is not None:
        pts = os.path.join(cfg["_work"], f"branch_{args.name}", "points")
        files = ([os.path.join(pts, f"state_{k:04d}.npz") for k in args.points] if args.points
                 else sorted(f for f in glob.glob(os.path.join(pts, "state_*.npz")) if not f.endswith("_vel.npz")))
    else:
        files = list(args.state or [])
    if not files:
        raise SystemExit("no states: give --state F [F ...] or --name B [--points K ...]")
    args.state = files[0]
    cfg, model, saved = _setup(args)
    model.diagnostics = list(dict.fromkeys(model.diagnostics + ["Velocity 1", "Velocity 2", "Velocity 3"]))
    spec = MapSpec(1e-12, 1, 1, 1.0)
    for f in files:
        out = os.path.splitext(f)[0] + "_vel.npz"
        if os.path.exists(out) and not args.force:
            log.info(f"velocity: {out} exists (--force to redo)")
            continue
        s = load_state(f)
        if s.layout.size != model.layout.size:
            log.warning(f"velocity: {f} has a different layout, skipped")
            continue
        model.frozen = s.frozen or model.frozen
        _, diag = model.run(s.x, s.p, spec, tag="velocity", want_diag=True)
        keep = {k: v for k, v in s.diag.items() if k not in diag}     # e.g. coordinates from older files
        save_state(out, s.layout, s.x, s.p, s.frozen, {**keep, **diag}, **s.meta)
        u = diag[find_name(diag, "velocity 1")][1]
        log.info(f"velocity: p = {s.p:g}, max |u| = {np.abs(u).max():.4g} m/yr; wrote {out}")


def cmd_toy_demo(args):
    """End-to-end run on the stand-in model (S-curve with an exact fold); no Elmer needed."""
    import tempfile
    work = args.work or tempfile.mkdtemp(prefix="ess_toy_")
    cfgp = os.path.join(work, "toy.toml")
    os.makedirs(work, exist_ok=True)
    with open(cfgp, "w") as f:
        f.write(f'''[paths]
work_dir = "{work}"
[model]
type = "toy"
[parameter]
name = "Ts"
value = -1.0
scale = 0.03
[map]
dt = 0.05
n_steps = 4
[stability_map]
dt = 0.005
n_steps = 20
[fd]
eps = 1e-6
[newton]
tol = 1e-10
[continuation]
ds = 0.3
dp0 = 0.05
ds_max = 1.0
ds_min = 1e-4
max_points = 40
p_min = -1.0
tol = 1e-9
stability_every = 5
stability_m = 8
[stability]
m = 10
nev = 4
''')
    ns = argparse.Namespace(config=cfgp, state=None, p=None, name=None, verbose=False)
    cmd_continue(ns)
    from .toy import ToyColumn
    m = ToyColumn()
    log.info(f"toy-demo: exact fold at Ts = {m.fold()[1]:.8f}; see {work}/branch_main/branch.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ess", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s, state=True):
        s.add_argument("config")
        s.add_argument("-v", "--verbose", action="store_true")
        if state:
            s.add_argument("--state", help="driver state file (.npz); default <work>/state0.npz")
            s.add_argument("--p", type=float, help="parameter value (default: the one stored with the state)")

    s = sub.add_parser("setup")
    common(s, state=False)
    s.add_argument("--ics-only", action="store_true", help="only rewrite the initial-condition tables")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("init")
    common(s, state=False)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("extract")
    common(s, state=False)
    s.add_argument("--restart", help="restart file name (relative to the mesh directory)")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("remask")
    common(s)
    s.add_argument("--out")
    s.add_argument("--all-cold", action="store_true", help="start from an all-cold mask (then switch nodes above T0)")
    s.set_defaults(fn=cmd_remask)

    s = sub.add_parser("check")
    common(s)
    s.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    s.add_argument("--skip-roundtrip", action="store_true")
    s.add_argument("--vector", choices=["random", "residual", "bed"], default="random",
                   help="direction of the linearity test: random over all nodes, the residual "
                        "Phi(x)-x (what Newton works with), or random basal temperatures only")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("run")
    common(s)
    s.add_argument("--chunks", type=int, default=1, help="number of map evaluations ([map] dt, n_steps each)")
    s.add_argument("--dt", type=float, help="time step (years), overrides [map] dt for this run")
    s.add_argument("--steps", type=int, help="steps per chunk, overrides [map] n_steps for this run")
    s.add_argument("--rock-cap", type=float, default=1.0,
                   help="factor on the bedrock heat capacity (default 1: physical transient)")
    s.add_argument("--remask", dest="remask", action="store_true", default=None,
                   help="move the frozen masks after every chunk (default: [masks] auto, on for marine)")
    s.add_argument("--no-remask", dest="remask", action="store_false")
    s.add_argument("--gl", choices=["contact", "frozen"], default="contact",
                   help="marine: grounding line by ElmerStreams contact every step (default, spin-ups), or the "
                        "frozen mask moved between chunks (as Newton)")
    s.add_argument("--save-every", type=int, default=0,
                   help="also save the state every N chunks to <out>_chunks/points/state_XXXX.npz (XXXX = chunk), "
                        "for plots/plot_branch_states.py; the --out file is rewritten after every chunk anyway")
    s.add_argument("--bed", choices=["temperate", "cold"],
                   help="the whole bed temperate or cold for this run (ColdTempMask +1 / -1 everywhere, not moved; "
                        "marine: the shelf base stays at T0 either way)")
    s.add_argument("--linear-stokes", choices=["direct", "iterative", "block"],
                   help="Stokes linear solver for this run: direct (MUMPS), iterative (GCR + ILU), block (block "
                        "preconditioning, keywords from [elmer] stokes_block_file); default: [elmer] linear_stokes")
    s.add_argument("--cartesian-bed", action="store_true",
                   help="test: the bed condition in x/y/z instead of normal-tangential components (nearly flat beds; "
                        "marine: with --gl frozen)")
    s.add_argument("--stokes-block-file", help="block-preconditioner keywords for --linear-stokes block "
                                                "(default [elmer] stokes_block_file, elmer/stokes_block.sif.in)")
    s.add_argument("--stokes-preconditioner", help="iterative Stokes preconditioner (e.g. ILU0, ILU8); "
                                                   "default [elmer] stokes_preconditioner")
    s.add_argument("--stokes-element", choices=["P2/P1", "Stabilized", "Bubbles"],
                   help="Stokes element for this run (default [elmer] stokes_element, P2/P1)")
    s.add_argument("--shelf-cut", type=float, metavar="KM",
                   help="EXPERIMENTAL: Stokes elements beyond x = KM passive (Elmer passive elements), the cut face "
                        "with a free-floating shelf's stress ([elmer] shelf_cut_km)")
    s.add_argument("--shelf-cut-no-target", action="store_true",
                   help="with --shelf-cut: no Passive Target boundary (stress-free cut face; to test the machinery)")
    s.add_argument("--output-level", type=int, help="Elmer's Max Output Level for this run ([elmer] output_level)")
    s.add_argument("--stokes-solver", choices=["nsvec", "flowsolve", "ins"],
                   help="Elmer's Stokes module: IncompressibleNSVec (default), FlowSolve, IncompressibleNS")
    s.add_argument("--rotate-guess", action="store_true",
                   help="rotate the Stokes initial guess to the normal-tangential frame before each linear solve "
                        "(elmer/RotateGuess.so; [elmer] stokes_rotate_guess)")
    s.add_argument("--add-diag", action="append", metavar="NAME",
                   help="also write this variable with the diagnostics (repeatable; e.g. Pressure for --warm-start)")
    s.add_argument("--warm-start", metavar="STATE",
                   help="start Stokes from the velocity (and pressure) in STATE's diagnostics (same mesh)")
    s.add_argument("--friction-heat", choices=["loads", "flux"],
                   help="frictional heating from the Stokes loads (nodal) or as a basal flux; default [elmer] friction_heat")
    s.add_argument("--chunk-offset", type=int, default=0,
                   help="number the --save-every states from this + 1 (continuing an interrupted run)")
    s.add_argument("--periodic-gl", choices=["none", "elmerstreams", "fix"],
                   help="contact grounding line with periodic sides: Elmer/Ice's, ElmerStreams' (Fw2) or the periodic "
                        "one (elmer/PeriodicGL.so); default [elmer] periodic_gl")
    s.add_argument("--lua", action="append", metavar="NAME=VALUE",
                   help="override a parameters.lua assignment for this run (repeatable), e.g. --lua fcoeff=0.002")
    s.add_argument("--stokes-only", action="store_true",
                   help="Stokes and the free surface only: no heat solve or strain heating (the temperature stays as in "
                        "the start state); for a bed whose sliding does not depend on temperature (with --bed)")
    s.add_argument("--heat-only", action="store_true",
                   help="no free surface: the geometry stays as in the start state, Stokes and heat run (with --dt 1e6 "
                        "--steps 2 --chunks 1: the steady temperature on that geometry in one run)")
    s.add_argument("--until-tol", type=float,
                   help="stop when the rms change of the (scaled) state over one chunk is below this; --chunks is "
                        "then the maximum")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("newton")
    common(s)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_newton)

    s = sub.add_parser("continue")
    common(s)
    s.add_argument("--name", help="branch name (directory <work>/branch_<name>)")
    s.add_argument("--ds", type=float, help="arclength step to (re)start with, overriding the saved one")
    s.add_argument("--temperate-k", type=int,
                   help="frozen temperate bed patch of this many columns from the terminus of the start state "
                        "(all else cold); each point gets the patch hydrology ([hydrology]) and thermal check")
    s.add_argument("--edge-thickness", type=float,
                   help="with --temperate-k: terminus = last column with ice thicker than this (m), as in spectrum")
    s.set_defaults(fn=cmd_continue)

    s = sub.add_parser("spectrum")
    common(s)
    s.add_argument("--k-max", type=int, help="largest temperate patch (bed columns from the terminus); default: to the divide")
    s.add_argument("--k-step", type=int, default=1, help="columns added per step (default 1)")
    s.add_argument("--tol-T", type=float, default=0.0, help="cold bed nodes may exceed T0 by this much (K)")
    s.add_argument("--p-step", type=float, help="sweep: also p + dp, p + 2 dp, ... (each from the previous all-cold state)")
    s.add_argument("--p-end", type=float, help="last parameter of the sweep (default [continuation] p_max)")
    s.add_argument("--stop-after-dry", type=int,
                   help="move on (next parameter) after this many consecutive failures (Newton or hydrology)")
    s.add_argument("--edge-thickness", type=float,
                   help="terminus = the last column (from the divide) with ice thicker than this (m), from the start "
                        "state; default: the frozen GroundedMask (the glaciated mask of the initial geometry)")
    s.add_argument("--maxit", type=int, help="Newton iterations per state (overrides [newton] maxit)")
    s.add_argument("--out", help="output directory (default <work>/spectrum_p<P>)")
    s.set_defaults(fn=cmd_spectrum)

    s = sub.add_parser("temperate-sweep")
    common(s)
    s.add_argument("--name", default="temperate", help="branch name (<work>/branch_<name>), default temperate")
    s.add_argument("--edge-thickness", type=float, default=50.0,
                   help="terminus = last column with ice thicker than this (m), default 50 (as the spectra)")
    s.add_argument("--chunks", type=int, default=0, help="transient spin-up chunks before Newton (default 0)")
    s.add_argument("--dt", type=float, help="spin-up / map time step (overrides [map] dt)")
    s.add_argument("--steps", type=int, help="steps per chunk (overrides [map] n_steps)")
    s.add_argument("--restart", action="store_true", help="converge the start state again even if it exists")
    s.add_argument("--spinup-tol", type=float,
                   help="spin up until the rms change of the (scaled) state over one chunk is below this, then "
                        "Newton and the sweep; --chunks is then the maximum (default 1000)")
    s.add_argument("--spinup-only", action="store_true",
                   help="only the transient chunks (to <work>/branch_<name>/spinup.npz), no Newton / sweep; "
                        "continue with --state that file")
    s.add_argument("--p-list", help="Tsurf values, comma separated")
    s.add_argument("--p-step", type=float, help="Tsurf from --p-start (default the start) to --p-end in this step")
    s.add_argument("--p-start", type=float)
    s.add_argument("--p-end", type=float)
    s.add_argument("--maxit", type=int, help="Newton iterations (overrides [newton] maxit)")
    s.add_argument("--heat-only", action="store_true",
                   help="per Tsurf: a steady heat solve on the frozen geometry (no free surface), checked by one "
                        "map evaluation; Newton only if that check fails")
    s.add_argument("--heat-dt", type=float, default=1e6, help="--heat-only: step length (yr), default 1e6")
    s.add_argument("--heat-steps", type=int, default=2, help="--heat-only: steps, default 2")
    s.set_defaults(fn=cmd_temperate_sweep)

    s = sub.add_parser("spectrum-down")
    common(s)
    s.add_argument("--k-step", type=int, default=5, help="temperate columns removed per step (default 5; cut "
                                                         "down to 1 where Newton fails)")
    s.add_argument("--k-min", type=int, default=0, help="stop at this patch size (default 0: all cold)")
    s.add_argument("--edge-thickness", type=float, default=50.0,
                   help="terminus = last column with ice thicker than this (m), from the start state (default 50)")
    s.add_argument("--tol-T", type=float, default=0.0, help="cold bed nodes may exceed T0 by this much (K)")
    s.add_argument("--maxit", type=int, help="Newton iterations (overrides [newton] maxit)")
    s.add_argument("--out", help="output folder (default <work>/spectrum_p<P>_e<edge>_down)")
    s.set_defaults(fn=cmd_spectrum_down)

    s = sub.add_parser("transfer")
    common(s)
    s.add_argument("--from", dest="source", required=True,
                   help="converged state to carry over (.npz, may come from another config)")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_transfer)

    s = sub.add_parser("stability")
    common(s)
    s.add_argument("--name", help="branch name, with --point")
    s.add_argument("--point", type=int, default=0)
    s.add_argument("--m", type=int, help="Krylov dimension (overrides [stability] m)")
    s.add_argument("--spanwise", action="store_true",
                   help="3D strip (y_elements > 1): project out the across-flow mean and classify the modes "
                        "by spanwise wavenumber")
    s.add_argument("--map", default="stability_map", choices=["stability_map", "map"],
                   help="map to linearise: [stability_map] (physical, default) or [map] (the one Newton "
                        "and continuation use: |mu| near 1 there means an ill-conditioned Newton system)")
    s.set_defaults(fn=cmd_stability)

    s = sub.add_parser("velocity")
    s.add_argument("config")
    s.add_argument("-v", "--verbose", action="store_true")
    s.add_argument("--state", nargs="+", help="state files (.npz)")
    s.add_argument("--name", help="branch name: its points (all, or --points)")
    s.add_argument("--points", type=int, nargs="+", help="branch point indices (default: all)")
    s.add_argument("--force", action="store_true", help="redo states that already have a _vel.npz")
    s.set_defaults(fn=cmd_velocity)

    s = sub.add_parser("toy-demo")
    s.add_argument("work", nargs="?")
    s.set_defaults(fn=cmd_toy_demo)

    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
