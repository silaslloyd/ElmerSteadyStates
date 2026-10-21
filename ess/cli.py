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
    return cfg, build_model(cfg, saved), saved


def _marine(model):
    return bool(getattr(model, "marine", False))


def _mask_cols(cfg, model, saved, x, diag):
    if cfg["model"].get("type", "elmer") != "elmer":
        return {}
    from .elmer import mask_report
    rep = mask_report(saved.layout, x, saved.frozen, diag, marine=_marine(model),
                      rho_ratio=getattr(model, "rho_ratio", None))
    if rep.get("cold_above_Tm") or rep.get("temp_freezing") or rep.get("to_grounded") or rep.get("to_floating"):
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
    prob = _problem(cfg, model)
    log.info(f"run: dt = {prob.spec.dt:g}, {prob.spec.n_steps} steps per chunk, "
             f"rock heat capacity x {prob.spec.rock_heat_capacity:g}"
             + (", grounding line by contact every step (--gl contact)" if contact else ""))
    p = saved.p if args.p is None else args.p
    y = prob.scale(saved.x)
    on, _ = _masks_auto(cfg, model)
    remask = on if args.remask is None else args.remask
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
    for c in range(args.chunks):
        yn = prob.phi(y, p, tag=f"run{c}", want_diag=True)
        t = (c + 1) * prob.spec.horizon
        log.info(f"run: chunk {c + 1}/{args.chunks}  t = {t:g}  rms change {rms(yn - y):.3e}")
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
        # the state after this chunk, with the masks it ran with (before the remask)
        save_state(out, saved.layout, prob.unscale(y), p, dict(saved.frozen), prob.last_diag, time=t)
        if args.save_every and (c + 1) % args.save_every == 0:
            save_state(os.path.join(sdir, f"state_{c + 1:04d}.npz"), saved.layout, prob.unscale(y), p,
                       dict(saved.frozen), prob.last_diag, time=t)
        if remask:
            xn, frozen, rep = update_masks(saved.layout, prob.unscale(y), saved.frozen, prob.last_diag,
                                           marine=_marine(model), rho_ratio=getattr(model, "rho_ratio", None),
                                           grounding=not contact)
            if n_changes(rep):
                log.info(f"run: remask after chunk {c + 1}: {describe(rep) if not contact else describe({k_: v_ for k_, v_ in rep.items() if not k_.startswith(('x_gl', 'to_gr', 'to_fl'))})}")
                saved.frozen.clear()
                saved.frozen.update(frozen)
                y = prob.scale(xn)
    save_state(out, saved.layout, prob.unscale(y), p, saved.frozen, prob.last_diag, time=args.chunks * prob.spec.horizon)
    log.info(f"wrote {out}")


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


def cmd_continue(args):
    cfg, model, saved = _setup(args)
    prob = _problem(cfg, model)
    cc = cfg.get("continuation", {})
    name = args.name or cc.get("name", "main")
    bdir = os.path.join(cfg["_work"], f"branch_{name}")
    every = int(cc.get("stability_every", 0))
    sopt = stability_options(cfg)
    sspec = map_spec(cfg, "stability_map")

    def on_save(k, y, p, row):
        x = prob.unscale(y)
        cols = _mask_cols(cfg, model, saved, x, prob.last_diag)
        extra = {}
        if every and k % every == 0:
            sopt_small = type(sopt)(**{**sopt.__dict__, "m": min(sopt.m, int(cc.get("stability_m", 12)))})
            st = stability(prob, y, p, sspec, sopt_small)
            extra = dict(n_unstable=st.n_unstable, sigma_max=float(np.max(st.sigma.real)),
                         mu_max=float(np.max(np.abs(st.mu))))
            cols.update(extra)
        save_state(os.path.join(bdir, "points", f"state_{k:04d}.npz"), saved.layout, x, p, saved.frozen,
                   prob.last_diag, s=row["s"], **{c: v for c, v in cols.items() if np.isscalar(v)})
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
    boundary) are discarded."""
    cfg, model, saved = _setup(args)
    if cfg["model"].get("type", "elmer") != "elmer":
        raise SystemExit("spectrum needs the Elmer model")
    from dataclasses import replace as _replace
    from .case import eval_lua_parameters
    from .spectrum import run_spectrum
    prob = _problem(cfg, model)
    p = saved.p if args.p is None else args.p
    P = eval_lua_parameters(model.params_file)
    out = args.out or os.path.join(cfg["_work"], f"spectrum_p{p:g}")
    nopt = newton_options(cfg)
    if args.maxit:
        nopt = _replace(nopt, maxit=args.maxit)
    rows = run_spectrum(prob, saved, prob.scale(saved.x), p, nopt, out, k_max=args.k_max, k_step=args.k_step,
                        tol_T=args.tol_T, rhoi=P["rhoi"], Lw=P["Lw"])
    kept = [r for r in rows if str(r["kept"]) == "1"]
    log.info(f"spectrum: {len(kept)} consistent states of {len(rows)} tried; {out}/spectrum.csv, {out}/points/")


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
    s.set_defaults(fn=cmd_continue)

    s = sub.add_parser("spectrum")
    common(s)
    s.add_argument("--k-max", type=int, help="largest temperate patch (bed columns from the terminus); default: to the divide")
    s.add_argument("--k-step", type=int, default=1, help="columns added per step (default 1)")
    s.add_argument("--tol-T", type=float, default=0.0, help="cold bed nodes may exceed T0 by this much (K)")
    s.add_argument("--maxit", type=int, help="Newton iterations per state (overrides [newton] maxit)")
    s.add_argument("--out", help="output directory (default <work>/spectrum_p<P>)")
    s.set_defaults(fn=cmd_spectrum)

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
