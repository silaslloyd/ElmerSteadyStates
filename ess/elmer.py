"""
The Elmer time-step map: x -> Phi(x) by one `mpirun ElmerSolver` run of elmer/steady_template.sif.

Each evaluation gets its own run directory under <work>/runs/. The driver writes
    parameters.lua   the case's parameters.lua with the continuation parameter overridden
    state_in.bin     the state (Temperature, Zs) and the frozen fields, merged, sorted by global node
    run.sif          the template with the placeholders filled in
runs Elmer there, and reads state_out.bin.<partition> back. Completed evaluations are cached by a
hash of (state, parameter, map settings, inputs), so a restarted job repeats no Elmer runs.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid

import numpy as np

from .model import MapSpec, Model, ModelFailure
from .statefile import (Records, StateLayout, find_name, merge_partitions, partition_files,
                        write_state_file)

log = logging.getLogger("ess")

PLACEHOLDER = re.compile(r"__[A-Z0-9_]+__")


def override_lua_assignment(text: str, name: str, value) -> str:
    """Replace the right-hand side of the first top-level `name = ...` in a Lua parameter file.
    Later lines that use `name` (derived quantities) then see the new value."""
    pat = re.compile(rf"^(\s*){re.escape(name)}(\s*)=(?!=)[^\n]*$", re.M)
    m = pat.search(text)
    if not m:
        raise KeyError(f"parameter '{name}' is not assigned in parameters.lua")
    line = f"{m.group(1)}{name} = {float(value)!r}   -- set by ElmerSteadyStates"
    return text[:m.start()] + line + text[m.end():]


class ElmerModel(Model):
    def __init__(self, cfg, layout: StateLayout | None = None, frozen: Records | None = None):
        self.cfg = cfg
        e = cfg["elmer"]
        self.repo = cfg["_repo"]
        self.work = cfg["_work"]
        self.case_dir = cfg["paths"]["case_dir"]
        # marine flowline ([case] geometry = "marine"): ice shelf, sea pressure, frozen grounded mask
        self.marine = cfg.get("case", {}).get("geometry", "land") == "marine"
        self.template = cfg["_path"](e.get("template", "elmer/steady_template_marine.sif" if self.marine
                                           else "elmer/steady_template.sif"))
        self.buoyancy_lib = cfg["_path"](e.get("buoyancy_lib", "elmer/Buoyancy"))
        # marine: Min Zb = bedrock (the floating base stops on the bed; without it, ice piling up at the
        # frozen grounding line pushed the base hundreds of metres through the bed in a spin-up)
        self.zb_limiter = bool(e.get("zb_limiter", True))
        # grounding line: "frozen" (mask from the state, smooth map: Newton, continuation) or "contact"
        # (ElmerStreams: moves every step; `ess run` uses it for spin-ups unless --gl frozen)
        self.gl_mode = str(e.get("gl_mode", "frozen"))
        # Stokes nonlinear iterations with the contact grounding line: SlidCoef_Contact ungrounds a node during the
        # Stokes solve (mask -1), and only a second iteration solves again without the zero normal velocity there.
        # With one iteration the base never leaves the bed and the grounding line cannot retreat (it still
        # advances: grounding is geometric). ElmerStreams Marine_Term: Nonlinear System Max Iterations = 2.
        self.contact_stokes_iters = int(e.get("contact_stokes_iterations", 2))
        # the contact sliding function without periodic sides: "elmerice" (Elmer/Ice's SlidCoef_Contact: the
        # ungrounding test only once Stokes has converged to Test Contact Tolerance) or "elmerstreams" (ElmerStreams'
        # contact.f90, here elmer/ContactPeriodic.F90 with Periodic Contact = False: the test after the first
        # iteration; needs PeriodicGL.so). Either tests from the second Stokes iteration of a step only.
        self.contact_function = str(e.get("contact_function", "elmerice"))
        # [elmer] sea_level_param = true: sea level is the Lua parameter ess_sl (m, default 0; e.g. --lua ess_sl=50)
        # in the constants (hydrostatic loads of the contact) and every material (sea pressure), not a fixed 0
        self.sea_level_param = bool(e.get("sea_level_param", False))
        self.mesh_db = e.get("mesh_db", "footprint")
        self.ess_lib = cfg["_path"](e.get("ess_lib", "elmer/SteadyStateIO"))
        # ESS_ELMER_LIBDIR=DIR: ESS's own Elmer libraries from DIR instead (e.g. elmer/checked: built with bounds checks)
        self._libdir = os.environ.get("ESS_ELMER_LIBDIR")
        if self._libdir:
            self.ess_lib = os.path.join(self._libdir, os.path.basename(self.ess_lib))
        # [elmer] stokes_rotate_guess: rotate the Stokes initial guess to the normal-tangential frame before every
        # (monolithic) linear solve (elmer/RotateGuess.F90: Elmer passes it unrotated, so warm starts - the previous
        # time step - begin far worse than a zero guess at every normal-tangential bed node)
        self.stokes_rotate_guess = bool(e.get("stokes_rotate_guess", False))
        self.ess_utils = cfg["_path"](e.get("ess_utils", "elmer/ess_utils.lua"))
        self.params_file = cfg["_path"](cfg.get("case", {}).get("parameters", "case/parameters.lua"))
        # ESS_NPROCS / ESS_MPIRUN (environment) override [elmer] n_procs / mpirun: several drivers in one batch
        # job, each with its own share of the cores (jobs/stokes_tests_*.cmd)
        self.n_procs = (int(os.environ.get("ESS_NPROCS", 0)) or int(e.get("n_procs", 0))
                        or int(os.environ.get("SLURM_NTASKS", "1")))
        self.mpirun = os.environ.get("ESS_MPIRUN") or e.get("mpirun", "mpirun -np {n}")
        self.solver = e.get("solver", "ElmerSolver")
        self.timeout = float(e.get("timeout", 6 * 3600))
        self.keep_runs = bool(e.get("keep_runs", False)) or os.environ.get("ESS_KEEP_RUNS") == "1"
        self.use_cache = bool(e.get("cache", True)) and os.environ.get("ESS_NO_CACHE") != "1"
        # warm start (ess run --warm-start): records (e.g. Velocity 1..3, Pressure) set by StateRead at the start
        # of the run, so the first Stokes solve starts from them instead of zero
        self.warm_records = {}
        self.diagnostics = list(e.get("diagnostics", []))
        self.extra_subs = dict(e.get("substitutions", {}))
        self.vtu = bool(e.get("vtu", False))
        self.friction_heat = str(e.get("friction_heat", "flux"))
        self.strain_heat = bool(e.get("strain_heat", False))
        # heat-only runs (`ess temperate-sweep --heat-only`): the free surface is not solved, so the geometry
        # stays as in the input state; Stokes still runs (same velocity every step on a frozen geometry
        # when nothing in the flow depends on temperature)
        self.heat_only = False
        # [elmer] cartesian_bed (test): the bed condition in x/y/z instead of normal-tangential (N-T) components:
        # vertical velocity 0 where grounded, Weertman sliding on u and v, the sea spring on w. For nearly flat beds
        # only (it takes the bed normal as vertical); for testing linear solvers whose N-T handling is in doubt
        # (Elmer's block solver before 2026-10-14). Frozen grounding line only (the contact one needs N-T).
        self.cartesian_bed = bool(e.get("cartesian_bed", False))
        # Elmer's output level ([elmer] output_level, default 3); e.g. 7 shows the block solver and its iterations
        self.output_level = int(e.get("output_level", 3))
        # Stokes-only runs (`ess run --stokes-only`): no heat solve and no strain heating, so the temperature
        # stays as in the input state; Stokes and the free surface run (the geometry with the flow, for a bed
        # whose sliding does not depend on temperature: fully temperate, or cold enough that the factor is capped)
        self.stokes_only = False
        # [elmer] min_zs (m above sea level): the surface limiter Min Zs = max(bed + 5, min_zs), so over a bed
        # below sea level the ice is at least as thick as the water is deep (a marine-like thickness);
        # unset: Min Zs = bed + 5 (the run inputs, and so the cached runs, are unchanged)
        mz = e.get("min_zs")
        self.min_zs_expr = "tx[0] + 5" if mz is None else f"max(tx[0] + 5, {float(mz)!r})"
        # surface temperature: "constant" (Tsurf everywhere) or "snout" (Tsurf where the ice is thicker
        # than snout_thickness, linear to T0 at zero thickness: ElmerStreams' warm snout)
        self.surface_temp = str(e.get("surface_temp", "constant"))
        # surface Robin condition k dT/dn = h (T_ext - T): h ("Heat transfer coefficient", ElmerStreams: 1)
        self.surface_transfer = float(e.get("surface_transfer", 1.0))
        # the kind of surface condition: "dirichlet" (T = the surface temperature) or "robin" (the transfer
        # condition above). Default: dirichlet on the marine template, robin on the land one
        self.surface_bc = str(e.get("surface_bc", "dirichlet" if self.marine else "robin"))
        if self.surface_bc not in ("dirichlet", "robin"):
            raise ValueError(f"[elmer] surface_bc must be 'dirichlet' or 'robin', not {self.surface_bc!r}")
        # bed under the unglaciated 30 m layer beyond the margin (GroundedMask 0): "flux" (geothermal +
        # friction, like the glaciated bed) or "melting" (T = T0, as ElmerStreams' tempbc)
        self.unglaciated_bed = str(e.get("unglaciated_bed", "flux"))
        if self.unglaciated_bed not in ("flux", "melting"):
            raise ValueError(f"[elmer] unglaciated_bed must be 'flux' or 'melting', not {self.unglaciated_bed!r}")
        if self.surface_temp not in ("constant", "snout"):
            raise ValueError(f"[elmer] surface_temp must be 'constant' or 'snout', not {self.surface_temp!r}")
        # linear solvers: "direct" (MUMPS) or "iterative" (Krylov + ILU, tolerance iterative_tol)
        self.linear_stokes = str(e.get("linear_stokes", "direct"))
        # "block": the keywords of this file go into the Stokes solver section verbatim (block preconditioning)
        self.stokes_block_file = str(e.get("stokes_block_file", "elmer/stokes_block.sif.in"))
        self.linear_heat = str(e.get("linear_heat", "direct"))
        # [elmer] mumps_workspace (percent): extra MUMPS workspace for the Stokes solve (Mumps Percentage Increase
        # Working Space); MUMPS stops with INFO(1) = -9 when its estimate is too small, e.g. as the grounding line
        # moves and the pivoting changes. Unset: MUMPS' default (the run inputs, and so the caches, unchanged)
        self.mumps_workspace = int(e.get("mumps_workspace", 0))
        # [elmer] linear_free_surface: "direct" (the template, default) or "iterative" (BiCGStab + ILU1)
        self.linear_free_surface = str(e.get("linear_free_surface", "direct"))
        self.iterative_tol = float(e.get("iterative_tol", 1e-10))
        self.stokes_precond = str(e.get("stokes_preconditioner", "ILU2"))
        self.ilut_tol = float(e.get("ilut_tolerance", 1e-4))     # drop tolerance for ILUT
        # "P2/P1" (default) or "Stabilized" (P1/P1 with stabilisation: far easier for ILU-preconditioned
        # Krylov solvers, but a different discretisation, so slightly different steady states)
        self.stokes_element = str(e.get("stokes_element", "P2/P1"))
        # [elmer] stokes_solver: Elmer's Stokes module: "nsvec" (IncompressibleNSVec, the default), "flowsolve" (the
        # classic FlowSolve that Elmer/Ice has long used) or "ins" (IncompressibleNS, NSVec's non-vectorised
        # predecessor). Same variable (Flow Solution[Velocity:3 Pressure:1]), boundary conditions and loads.
        self.stokes_solver = str(e.get("stokes_solver", "nsvec"))
        # [elmer] sea_spring_dt (years, marine): the time step in the sea-pressure spring at the shelf base and front
        # (Buoyancy SeaSpring, rho_w g dt n; Elmer/Ice's "Sea Spring Timestep Size"), fixed instead of the run's dt.
        # Tiny steps (e.g. 1e-4 yr at a run's start) make that spring ~0 and Stokes nearly singular under floating
        # ice; a fixed spring keeps them well-posed but damps the free surface more in those steps. Env
        # ESS_SEA_SPRING_DT overrides (tests). Unset: the run's dt (as before)
        ssd = os.environ.get("ESS_SEA_SPRING_DT") or e.get("sea_spring_dt")
        self.sea_spring_dt = None if ssd in (None, "") else float(ssd)
        # EXPERIMENTAL [elmer] shelf_cut_km (marine): Elmer's passive elements (as Elmer/Ice FS-SSA coupling): the ice
        # elements beyond x = shelf_cut_km are passive for Stokes; shelf_cut_target (default true): the face between
        # active and passive elements is a "Passive Target" boundary with a free-floating shelf's stress (see
        # _shelf_cut). Crashed in the first tests (jobs/passive_tests.cmd diagnoses it); jobs/scaling_tests.cmd cuts
        # the mesh instead ([case] front_km).
        sc = e.get("shelf_cut_km")
        self.shelf_cut_x = None if sc is None else float(sc) * 1e3
        self.shelf_cut_target = bool(e.get("shelf_cut_target", True))
        # [elmer] fs_ssa_km (marine; env ESS_FS_SSA_KM overrides): a MOVING full-Stokes region. Every time step the
        # horizontal distance of every node to the current grounding line is computed (elmer/GLDistance.F90, from
        # GroundedMask); ice more than fs_ssa_km beyond it is passive for Stokes, and the face between active and
        # passive ice carries the normal stress of a free-floating, laterally uniform shelf (as shelf_cut_target:
        # exact for an unbuttressed shelf, e.g. between free-slip walls). STAGE 1: the passive shelf keeps its last
        # velocities (no SSA solve yet), so its geometry evolves with them.
        fs = os.environ.get("ESS_FS_SSA_KM") or e.get("fs_ssa_km")
        self.fs_ssa_x = None if fs in (None, "") else float(fs) * 1e3
        # [elmer] fs_ssa_shelf (env ESS_FS_SSA_SHELF): "frozen" (default: stage 1 above) or "ssa": the passive shelf
        # solved by the shallow-shelf approximation every step (GLDistance.F90 FSSSAShelf: inflow from the
        # depth-averaged Stokes velocity on the interface, natural calving front, linear viscous ice), kept in
        # flotation and moved with that velocity, and the interface stress taken from the SSA strain rate instead of
        # the free-floating formula (FSSSALinsolve, "FS SSA Shelf Stress"). Side walls: v = 0 unless periodic_sides.
        self.fs_ssa_shelf = (os.environ.get("ESS_FS_SSA_SHELF") or e.get("fs_ssa_shelf") or "frozen").lower()
        if self.fs_ssa_shelf not in ("frozen", "ssa"):
            raise ValueError(f"[elmer] fs_ssa_shelf must be frozen or ssa, not {self.fs_ssa_shelf!r}")
        # iterative Stokes ([elmer] linear_stokes = "iterative"): tolerance (default iterative_tol) and whether to
        # stop when GCR does not converge (default yes; ElmerStreams' linsys/gcr.sif: 1e-8 and no)
        self.stokes_iterative_tol = e.get("stokes_iterative_tol")
        self.stokes_abort = bool(e.get("stokes_abort_not_converged", True))
        # [lua]: extra global assignments, appended to parameters.lua (they override ess_utils.lua
        # defaults such as ess_Hs / ess_Hmin, which is loaded first)
        self.lua_overrides = {k: repr(float(v)) for k, v in cfg.get("lua", {}).items()}
        # fixed-margin mode: the domain ends at x_end * L (the steady margin, 0.8 L), Zs clamped
        # to bed + edge_thickness there, plain mass balance (no thickness factor)
        c = cfg.get("case", {})
        self.x_end = float(c.get("x_end", 1.0))
        if self.x_end < 1.0:
            self.lua_overrides = {"ess_xend": f"{self.x_end!r} * L",
                                  "ess_Hedge": repr(float(c.get("edge_thickness", 30.0))),
                                  "ess_Hs": "0.0", **self.lua_overrides}
        if self.marine:
            self.lua_overrides = {"ess_marine": "1.0", **self.lua_overrides}
        if self.unglaciated_bed == "melting":
            self.lua_overrides = {"ess_ungl_melt": "1.0", **self.lua_overrides}
        if self.surface_temp == "snout":
            self.lua_overrides = {"ess_snoutH": repr(float(e.get("snout_thickness", 100.0))),
                                  "ess_snoutW": repr(float(e.get("snout_smooth", 0.0))),
                                  **self.lua_overrides}
        self.state_vars = list(cfg["state"]["variables"])
        self.frozen_vars = list(cfg["state"].get("frozen", []))
        self.param_name = cfg["parameter"]["name"]
        self.layout = layout
        self.frozen = frozen or {}
        self.nrun = 0
        os.makedirs(os.path.join(self.work, "runs"), exist_ok=True)
        os.makedirs(os.path.join(self.work, "cache"), exist_ok=True)
        with open(self.template) as f:
            self._template_text = f.read()
        with open(self.params_file) as f:
            self._params_text = f.read()
        from .case import eval_lua_parameters
        P = eval_lua_parameters(self.params_file)
        self.rho_ratio = float(P["rhoi"] / P["rhow"]) if "rhoi" in P and "rhow" in P else 920 / 1003.27
        # [case] periodic_sides (marine template): side 2 the periodic image of side 1 for the bedrock, filler and
        # ice (as ElmerStreams' periodic template), no Velocity 2 = 0 on the ice sides; translation [case] width
        # (else parameters.lua Wmesh). The mesh needs ElmerGrid -periodic 0 1 0 (`ess setup` adds it).
        self.periodic_sides = bool(c.get("periodic_sides", False))
        self.width = float(c.get("width", P.get("Wmesh", 1000.0)))
        # [case] slide_ybump: the sliding coefficient times 1 + slide_ybump sin(2 pi y / width) (periodic), so the
        # grounding line is not straight across the flow (tests of the periodic sides); 0: off
        self.slide_ybump = float(c.get("slide_ybump", 0.0))
        # [elmer] periodic_gl (contact grounding line with periodic sides; `ess run --periodic-gl`):
        #   "none"          Elmer/Ice SlidCoef_Contact (hydrostatic loads Fw)
        #   "elmerstreams"  ElmerStreams' contact.f90: the original test on the raw Stokes loads, with the
        #                   hydrostatic loads doubled on the master side (Fw2, FixFw)
        #   "fix"           SlidCoef_ContactPeriodic: periodic loads, hydrostatic loads, normals and grounded mask
        #                   in the test, and the grounded mask made periodic after GroundedSolver
        # (elmer/PeriodicGL.so: make -C elmer PeriodicGL.so)
        self.periodic_gl = str(e.get("periodic_gl", "none"))
        # [elmer] thermal = "elmerstreams" (marine template, contact grounding line): ElmerStreams Marine_Term's
        # thermal chain in place of the frozen cold/temperate mask: the mask set every time step by its
        # ColdTempMaskSolver (cold -> temperate above the melting point, temperate -> cold when the water sheet
        # is gone), the SheetSolverhw water sheet fed by the melt from the heat solver's loads (getmeltrate,
        # effective pressure from Stress 3: ComputeDevStress), loads-based friction heating and strain heating.
        # With periodic_gl "fix" the loads are periodic copies (FixPeriodicLoads) and nothing is halved; with
        # "elmerstreams" the friction heating and melt are halved on the y < 1 side. Needs elmer/ColdTempMask
        # and elmer/SheetSolverhw (make -C elmer thermal), friction_heat = "loads", strain_heat = true, and
        # "Water Sheet Thickness" among the [state] frozen fields (carried from chunk to chunk).
        self.thermal = str(e.get("thermal", "frozen"))
        self.thermal_libs = {k: cfg["_path"](e.get(f"{k}_lib", f"elmer/{v}")) for k, v in
                             (("coldtemp", "ColdTempMask"), ("sheet", "SheetSolverhw"))}
        self.hydro_lua = cfg["_path"](e.get("hydro_lua", "elmer/ess_hydro.lua"))
        # thermal = "loads": only the friction heating from the Stokes loads made periodic-aware (fix: from the
        # periodic copy, run by the heat solver first; elmerstreams: halved at y < 1; none: raw), the frozen
        # cold/temperate mask as it is (e.g. all cold: the grounded bed free, no temperature cap)
        # [elmer] sliding: "subtemperate" (TDepSliding, the default) or "temperate" (fcoeff everywhere, as at T0)
        self.sliding = str(os.environ.get("ESS_SLIDING") or e.get("sliding", "subtemperate"))   # env: tests
        self.periodic_lib = cfg["_path"](e.get("periodic_lib", "elmer/PeriodicGL"))
        if self._libdir:                     # ESS_ELMER_LIBDIR (see ess_lib): all of ESS's own libraries from there
            self.buoyancy_lib = os.path.join(self._libdir, os.path.basename(self.buoyancy_lib))
            self.periodic_lib = os.path.join(self._libdir, os.path.basename(self.periodic_lib))
        self._check_inputs()

    # ---------------------------------------------------------------- set-up checks
    def _check_inputs(self):
        mesh = os.path.join(self.case_dir, self.mesh_db)
        if len(mesh) > 127:      # ElmerSolver truncates the mesh path and then crashes reading a partition
            raise ValueError(f"mesh path {mesh} is {len(mesh)} characters: Elmer only reads 128 (it cuts the rest "
                             f"off and fails to find the mesh). Use a shorter work_dir / case_dir.")
        if not os.path.isdir(mesh):
            log.warning(f"mesh directory not found: {mesh} (run `ess setup` first)")
        elif self.n_procs > 1 and not os.path.isdir(os.path.join(mesh, f"partitioning.{self.n_procs}")):
            log.warning(f"no {mesh}/partitioning.{self.n_procs}: partition the mesh for {self.n_procs} "
                        f"processes (ElmerGrid ... -metiskway {self.n_procs})")
        for l in [self.ess_lib] + ([self.buoyancy_lib] if self.marine else []):
            lib = l if l.endswith(".so") else l + ".so"
            if not os.path.exists(lib):
                log.warning(f"{lib} not found: build it with `make -C elmer` (needs elmerf90)")

    # ---------------------------------------------------------------- inputs of a run
    def params_lua(self, p) -> str:
        txt = self._params_text
        # [lua] overrides of parameters.lua assignments replace them in place, so derived quantities
        # (e.g. dT = delta*Tscale) follow; other names (ess_utils.lua globals) are appended
        extra = {}
        for k, v in self.lua_overrides.items():
            pat = re.compile(rf"^(\s*){re.escape(k)}(\s*)=(?!=)[^\n]*$", re.M)
            m = pat.search(txt)
            if m and k != self.param_name:
                txt = txt[:m.start()] + f"{m.group(1)}{k} = {v}   -- [lua] in the ess config" + txt[m.end():]
            else:
                extra[k] = v
        txt = override_lua_assignment(txt, self.param_name, p)
        if getattr(self, "sea_level_param", False) and "ess_sl" not in self.lua_overrides:
            extra["ess_sl"] = "0.0"                     # sea level (m) of sea_level_param, unless overridden
        if extra:
            txt += "\n-- [lua] overrides from the ess config\n" + "".join(f"{k} = {v}\n" for k, v in extra.items())
        return txt

    def _friction_heat_block(self) -> str:
        """Frictional heating at the bed (boundary condition 19).
        loads: the Stokes reaction loads times the velocity, as nodal heat loads (ElmerStreams). Nodal
               loads are integrated quantities shared out by the elements; on this mesh they come out
               uneven from node to node, which with the sharp temperature-dependent sliding law seeds
               a node-to-node oscillation of the basal temperature.
        flux:  a heat flux tau_b . u_b = C(T) |u_b|^2 from the same linear Weertman law (per unit area,
               smooth from node to node); the geothermal flux comes through the bedrock as before."""
        if self.friction_heat == "loads":
            return ('  Temperature Load = Variable Flow Solution Loads 1, Flow Solution Loads 2, '
                    'Flow Solution Loads 3, Velocity 1, Velocity 2, Velocity 3, Coordinate 2\n'
                    '    Real Lua "frictionloads(tx[0],tx[1],tx[2],tx[3],tx[4],tx[5], W, W)"\n'
                    '  Temperature Load Condition = Equals "GroundedMask"')
        if self.friction_heat == "flux":
            return ('  Heat Flux BC = Logical True\n'
                    '  Heat Flux = Variable Temperature, Velocity 1, Velocity 2, Velocity 3, Coordinate 1\n'
                    '    Real Lua "TDepSliding(fcoeff, tx[0], dT, tx[4]) * (tx[1]*tx[1] + tx[2]*tx[2] + tx[3]*tx[3])"')
        raise ValueError(f"[elmer] friction_heat must be 'loads' or 'flux', not {self.friction_heat!r}")

    @staticmethod
    def _dt_ramp(text: str, ramp: str, n_steps: int) -> str:
        """TESTING (solver benchmarks, jobs/scaling_tests.cmd DT_RAMP): replace the uniform time step by a sequence,
        env ESS_DT_RAMP = "1e-4:1e-2:0.1*3" (years; "dt*n" = n steps of dt) as Elmer timestep intervals. The steps
        must add up to the run's step count; ESS's own time bookkeeping still assumes the uniform dt."""
        sizes, counts = [], []
        for item in [i for i in re.split(r"[,:;\s]+", ramp) if i]:   # ":" for sbatch --export (splits at commas)
            dt, _, n = item.partition("*")
            sizes.append(float(dt)); counts.append(int(n) if n else 1)
        if sum(counts) != n_steps:
            raise ValueError(f"ESS_DT_RAMP {ramp!r} has {sum(counts)} steps, the run {n_steps}")
        k = len(sizes)
        new = (f"  Timestep Intervals({k}) = Integer {' '.join(map(str, counts))}\n"
               f"  Timestep Sizes({k}) = Real {' '.join(repr(x) for x in sizes)}\n"
               f"  Output Intervals({k}) = Integer {' '.join(map(str, counts))}   ! ESS_DT_RAMP")
        text, n = re.subn(r"^ *Timestep Intervals = .*\n *Timestep Sizes = .*\n *Output Intervals = .*$", new,
                          text, count=1, flags=re.M)
        if n != 1:
            raise ValueError("ESS_DT_RAMP: Timestep Intervals / Sizes / Output Intervals lines not found in the template")
        return text

    def _stokes_linear(self) -> str:
        """the Stokes linear-solver keywords, with the initial-guess rotation if asked (not for the block solver)"""
        txt = self._linear_block(self.linear_stokes, "stokes")
        if self.stokes_rotate_guess and not re.search(r'Linear System Solver\s*=\s*"?Block', txt, re.I):
            lib = os.path.join(os.path.dirname(self.ess_lib), "RotateGuess")
            txt += f'\n  Before Linsolve = File "{lib}" "RotateInitialGuess"   ! [elmer] stokes_rotate_guess'
        return txt

    def _linear_block(self, kind, which) -> str:
        """Linear-solver keywords for one solver. Iterative solves must converge far below the finite-
        difference perturbation (fd eps): their error is noise in the Jacobian products (check with
        `ess check`: the repeat difference rms(Phi_a - Phi_b) and the linearity plateau)."""
        if kind == "direct":
            ws = (f'\n  Mumps Percentage Increase Working Space = Integer {self.mumps_workspace}'
                  if self.mumps_workspace and which == "stokes" else "")
            return ('  Linear System Solver = Direct\n'
                    '  Linear System Direct Method = $directsolver\n'
                    '  Linear System Abort Not Converged = True\n'
                    '  Linear System Convergence Tolerance = 1.0e-12' + ws)
        if kind == "block" and which == "stokes":
            path = self.stokes_block_file
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
            with open(path) as f:
                lines = [l.rstrip() for l in f if l.strip() and not l.lstrip().startswith(("#", "//"))]
            return "\n".join("  " + l.lstrip() for l in lines)
        if kind != "iterative":
            raise ValueError(f"[elmer] linear solver must be 'direct', 'iterative' or 'block' (Stokes), not {kind!r}")
        tol = f"{self.iterative_tol:.1e}"
        if which == "stokes" and self.stokes_iterative_tol is not None:
            tol = f"{float(self.stokes_iterative_tol):.1e}"
        if which == "stokes":
            # saddle-point system: GCR with a long restart and a strong ILU
            return ('  Linear System Solver = Iterative\n'
                    '  Linear System Iterative Method = GCR\n'
                    '  Linear System GCR Restart = Integer 250\n'
                    '  Linear System Max Iterations = 5000\n'
                    f'  Linear System Preconditioning = {self.stokes_precond}\n'
                    + (f'  Linear System ILUT Tolerance = Real {self.ilut_tol:.1e}\n'
                       if self.stokes_precond.upper() == "ILUT" else '') +
                    f'  Linear System Convergence Tolerance = {tol}\n'
                    f'  Linear System Abort Not Converged = {"True" if self.stokes_abort else "False"}\n'
                    '  Linear System Residual Output = 200')
        # heat (advection-diffusion, non-symmetric) and strain heating (mass matrix)
        return ('  Linear System Solver = Iterative\n'
                '  Linear System Iterative Method = BiCGStabL\n'
                '  Linear System BiCGStabL Polynomial Degree = Integer 4\n'
                '  Linear System Max Iterations = 2000\n'
                '  Linear System Preconditioning = ILU1\n'
                f'  Linear System Convergence Tolerance = {tol}\n'
                '  Linear System Abort Not Converged = True\n'
                '  Linear System Residual Output = 200')

    def _gl_blocks(self) -> dict:
        """marine template: the grounding-line treatment (frozen mask, or ElmerStreams contact)"""
        contact = self.gl_mode == "contact"
        if self.gl_mode not in ("frozen", "contact"):
            raise ValueError(f"[elmer] gl_mode must be 'frozen' or 'contact', not {self.gl_mode!r}")
        weert = ('  Slip Coefficient 2 = Variable Coordinate 1\n    Real Procedure "ElmerIceUSF" "{f}"\n'
                 '  Slip Coefficient 3 = Variable Coordinate 1\n    Real Procedure "ElmerIceUSF" "{f}"')
        if not contact:
            return {"__EQ4_SOLVERS__": "(2) = 8 12",
                    "__NOOP_EXPORTS__": '  Exported Variable 1 = -dofs 1 "GroundedMask"\n'
                                        '  Exported Variable 2 = -dofs 1 "ColdTempMask"',
                    "__GL_SOLVERS__": "",
                    "__GL_SLIDING__": weert.format(f="Sliding_Weertman")}
        if self.periodic_gl not in ("none", "elmerstreams", "fix"):
            raise ValueError(f"[elmer] periodic_gl must be 'none', 'elmerstreams' or 'fix', not {self.periodic_gl!r}")
        extra, eq4, sliding = "", "(4) = 8 12 15 16", weert.format(f="SlidCoef_Contact")
        if self.contact_function not in ("elmerice", "elmerstreams"):
            raise ValueError(f"[elmer] contact_function must be 'elmerice' or 'elmerstreams', not {self.contact_function!r}")
        if self.periodic_gl == "none" and self.contact_function == "elmerstreams":
            sliding = (weert.format(f="SlidCoef_ContactPeriodic").replace('"ElmerIceUSF"', f'"{self.periodic_lib}"')
                       + '\n  Periodic Contact = Logical False\n  Hydrostatic Loads Variable = String "Fw"')
        if self.periodic_gl != "none":
            lib = self.periodic_lib
            sliding = (weert.format(f="SlidCoef_ContactPeriodic").replace('"ElmerIceUSF"', f'"{lib}"')
                       + ("\n  Periodic Contact = Logical False\n  Hydrostatic Loads Variable = String \"Fw2\""
                          if self.periodic_gl == "elmerstreams" else
                          "\n  Periodic Contact = Logical True\n  Hydrostatic Loads Variable = String \"Fw\""))
            eq4 = "(5) = 8 12 15 16 17"
            if self.periodic_gl == "elmerstreams":
                extra = f"""
! ElmerStreams' FixFw: the hydrostatic loads doubled on the master side (the periodic root gets the pair sum)
Solver 17
  Exec Solver = "Before Timestep"
  Equation = "Fw2"
  Procedure = "{lib}" "FixPeriodicLoads"
  Source Variable = String "Fw"
  Target Variable = String "Fw2"
  Periodic Mode = String "fold"
End
"""
            else:
                extra = f"""
! the grounded mask the same on both periodic copies (GroundedSolver's grounding-line pass is one-sided)
Solver 17
  Exec Solver = "After Timestep"
  Equation = "Periodic GroundedMask"
  Procedure = "{lib}" "FixPeriodicLoads"
  Source Variable = String "GroundedMask"
  Target Variable = String "GroundedMask"
  Periodic Mode = String "min"
End
"""
        return {"__EQ4_SOLVERS__": eq4,
                "__NOOP_EXPORTS__": '  Exported Variable 1 = -dofs 1 "ColdTempMask"',
                "__GL_SOLVERS__": """! ElmerStreams grounding line (gl_mode = contact): hydrostatic loads for the contact test, and the
! grounded mask from Zb and the bed after every step (SlidCoef_Contact also ungrounds nodes in Stokes)
Solver 15
  Exec Solver = "Before Timestep"
  Equation = "Fw"
  Procedure = "ElmerIceSolvers" "GetHydrostaticLoads"
  Variable = Fw[Fwater:3]
  Variable DOFs = 3
End

Solver 16
  Exec Solver = "After Timestep"
  Equation = "GroundedMask"
  Procedure = "ElmerIceSolvers" "GroundedSolver"
  Variable = "GroundedMask"
  Variable DOFs = 1
  Toler = Real 1.0e-3
  Bedrock Variable = String "bedrock"
  Bubbles in global system = False
End
""" + extra,
                "__GL_SLIDING__": sliding + """
  Grounding Line Moves = Logical True
  Grounding Line Definition = String "Discontinuous"
  Test Contact Tolerance = Real 1.0E-3"""}

    def _write_vars(self, names) -> str:
        return "\n".join(f'  State Variable {i + 1} = String "{n}"' for i, n in enumerate(names))

    def build_sif(self, spec: MapSpec, *, state_in="state_in.bin", state_out="state_out.bin",
                  state_out0="none", write0=(), restart=None, restart_before_ics=True, runname="ess") -> str:
        out_vars = list(dict.fromkeys(self.state_vars + self.diagnostics))
        rb = ""
        if restart:
            rb = (f'  Restart File = "{restart}"\n  Restart Position = 0\n'
                  f'  Restart Before Initial Conditions = Logical {"True" if restart_before_ics else "False"}\n'
                  f'  Restart Time Ignore = Logical True')
        pb = f'  Post File = "{runname}.vtu"' if self.vtu else ""
        subs = {
            "__RUNNAME__": runname,
            "__ESS_UTILS__": self.ess_utils,
            "__PARAMS__": "parameters.lua",
            "__MESH_DIR__": self.case_dir,
            "__MESH_NAME__": self.mesh_db,
            "__CASE_DIR__": self.case_dir,
            "__ESS_LIB__": self.ess_lib,
            "__BUOYANCY_LIB__": self.buoyancy_lib,
            **self._gl_blocks(),
            "__ZB_MIN__": ('  Min Zb = Equals "bedrock"' if self.zb_limiter
                           else "  ! no Min Zb ([elmer] zb_limiter = false)"),
            "__NSTEPS__": str(int(spec.n_steps)),
            "__DT__": repr(float(spec.dt)),
            "__COUPLED_ITERS__": str(int(spec.coupled_iters)),
            "__ROCK_CAP__": repr(float(spec.rock_heat_capacity)),
            "__STATE_IN__": state_in,
            "__STATE_OUT__": state_out,
            "__STATE_OUT0__": state_out0,
            "__WRITE_VARS__": self._write_vars(out_vars),
            "__WRITE_VARS0__": self._write_vars(list(write0) or self.state_vars),
            "__RESTART_BLOCK__": rb,
            "__POST_BLOCK__": pb,
            "__FRICTION_HEAT__": self._friction_heat_block(),
            "__FS_EXEC__": '"Never"' if self.heat_only else '"after Timestep"',
            "__OUTPUT_LEVEL__": str(self.output_level),
            "__MIN_ZS_EXPR__": self.min_zs_expr,
            "__LS_STOKES__": self._stokes_linear(),
            "__LS_HEAT__": self._linear_block(self.linear_heat, "heat"),
            "__LS_STRAIN__": self._linear_block(self.linear_heat, "strain"),
            # "Bubbles": ElmerStreams' P1 + bubbles (Stabilization Method = Bubbles with its Element line)
            "__STOKES_STAB__": ('"Bubbles"\n  Element = "p:1 -tri b:1 -quad b:3 -wedge b:1 -brick b:4"'
                                if self.stokes_element == "Bubbles" else f'"{self.stokes_element}"'),
            # strain heating (ElmerStreams: on): DeformationalHeatSolver + volumetric heat source
            "__STRAIN_EXEC__": '"Always"' if self.strain_heat and not self.stokes_only else '"Never"',
            "__HEAT_EXEC__": '\n  Exec Solver = "Never"   ! --stokes-only' if self.stokes_only else "",
            "__HEAT_SOURCE__": ('  Heat Source = Variable Strain Heating\n    Real Lua "tx[0]/rhoi"'
                                if self.strain_heat else "  ! no Heat Source: strain heating off ([elmer] strain_heat)"),
            "__HTC__": f"{self.surface_transfer:g}",
            "__SURFACE_T__": ('  External Temperature = Variable height\n'
                              '    Real Lua "surfaceT(tx[0] - 10*zscale, Tsurf)"'
                              if self.surface_temp == "snout" else '  External Temperature = Real #Tsurf'),
            # fixed-margin mode: Zs = bed + edge thickness on the end of the domain (Dirichlet)
            "__FRONT_ZS__": ('  Zs = Variable "bedrock"\n    Real Lua "tx[0] + ess_Hedge"'
                             if self.x_end < 1.0 else ""),
        }
        subs.update({k if k.startswith("__") else f"__{k}__": str(v) for k, v in self.extra_subs.items()})
        text = self._template_text
        for k, v in subs.items():
            text = text.replace(k, v)
        if self.sea_spring_dt is not None:
            text, n = re.subn(r"^Constants *\n", "Constants\n  Sea Spring Timestep Size = Real "
                              f"{self.sea_spring_dt!r}   ! [elmer] sea_spring_dt\n", text, count=1, flags=re.M)
            if n != 1:
                raise ValueError("[elmer] sea_spring_dt: no Constants section in the template")
        nl = os.environ.get("ESS_STOKES_NL_ITERS")      # TESTING: Stokes nonlinear iterations per step (ElmerStreams
        if nl:                                          # marine: 2, tolerance 1e-5); the template has 1
            i0 = text.find('"IncompressibleNSVec"')
            i1 = text.find("\nEnd", i0)
            if i0 < 0 or i1 < 0:
                raise ValueError("ESS_STOKES_NL_ITERS: no IncompressibleNSVec section")
            sec, n = re.subn(r"Nonlinear System Max Iterations = \d+", f"Nonlinear System Max Iterations = {int(nl)}",
                             text[i0:i1], count=1)
            sec = re.sub(r"Nonlinear System Convergence Tolerance *= *[0-9.eE+-]+",
                         "Nonlinear System Convergence Tolerance = 1.0e-5", sec, count=1)
            if n != 1:
                raise ValueError("ESS_STOKES_NL_ITERS: no Nonlinear System Max Iterations in the Stokes section")
            text = text[:i0] + sec + text[i1:]
        ramp = os.environ.get("ESS_DT_RAMP")
        if ramp:
            text = self._dt_ramp(text, ramp, int(spec.n_steps))
        if self.cartesian_bed:
            text = self._cartesian_bed(text)
        if self.periodic_sides:
            text = self._periodic_sides(text)
        if self.linear_free_surface == "iterative":
            text = self._fs_iterative(text)
        if self.surface_bc == "dirichlet":
            text = self._surface_dirichlet(text)
        if self.marine and self.sea_level_param:
            text = self._sea_level_param(text)
        # (ESS_STOKES_NL_ITERS, set above, takes precedence: ElmerStreams' max n with a 1e-5 tolerance)
        if (self.marine and self.gl_mode == "contact" and self.contact_stokes_iters > 1
                and not os.environ.get("ESS_STOKES_NL_ITERS")):
            text = self._stokes_iterations(text, self.contact_stokes_iters)
        if self.shelf_cut_x is not None:
            text = self._shelf_cut(text)
        if self.fs_ssa_x is not None:
            text = self._fs_ssa(text)
        if self.stokes_solver != "nsvec":
            procs = {"flowsolve": '"FlowSolve" "FlowSolver"', "ins": '"IncompressibleNS" "IncompressibleNSSolver"'}
            if self.stokes_solver not in procs:
                raise ValueError(f"[elmer] stokes_solver must be nsvec, flowsolve or ins, not {self.stokes_solver!r}")
            old = 'Procedure = "IncompressibleNSVec" "IncompressibleNSSolver"'
            if text.count(old) != 1:
                raise ValueError("stokes_solver: the Stokes procedure line of the template changed")
            text = text.replace(old, f"Procedure = {procs[self.stokes_solver]}   ! [elmer] stokes_solver = {self.stokes_solver}")
        if self.slide_ybump:
            text = self._slide_ybump(text)
        if self.thermal == "elmerstreams":
            text = self._thermal_elmerstreams(text)
        elif self.thermal == "loads":
            text = self._thermal_loads(text)
        elif self.thermal != "frozen":
            raise ValueError(f"[elmer] thermal must be 'frozen', 'loads' or 'elmerstreams', not {self.thermal!r}")
        if self.sliding == "temperate":
            text = self._sliding_temperate(text)
        elif self.sliding != "subtemperate":
            raise ValueError(f"[elmer] sliding must be 'subtemperate' or 'temperate', not {self.sliding!r}")
        left = sorted(set(PLACEHOLDER.findall(text)))
        if left:
            raise ValueError(f"unfilled placeholders in {self.template}: {', '.join(left)}")
        return text

    def _periodic_sides(self, text):
        """marine template: side 2 the periodic image of side 1 (bedrock, filler, ice), free ice sides"""
        if not self.marine:
            raise ValueError("[case] periodic_sides: marine template only")
        per = lambda master, names: (f"  Periodic BC = {master}\n"
                                     f"  Periodic BC Translate(3) = Real 0.0 {self.width!r} 0.0\n"
                                     + "".join(f"  Periodic BC {n} = Logical True\n" for n in names))
        ice = ["Velocity 1", "Velocity 2", "Velocity 3", "Pressure", "Temperature", "Zs", "Zb"]
        if self.thermal == "elmerstreams":
            ice.append("Water Sheet Thickness")
        rock = ["Temperature"]
        for name, master, names in (("side2 bedrock", 7, rock), ("side2 filler", 11, rock), ("side2 ice", 15, ice)):
            k = text.index(f'Name = "{name}"')
            j = text.index("\nEnd", k)
            text = text[:j + 1] + per(master, names) + text[j + 1:]
        for name in ("side1 ice", "side2 ice"):
            k = text.index(f'Name = "{name}"')
            j = text.index("\nEnd", k)
            block = text[k:j]
            if "  Velocity 2 = 0.0\n" not in block:
                raise ValueError(f"periodic_sides: no 'Velocity 2 = 0.0' in {name}")
            text = text[:k] + block.replace("  Velocity 2 = 0.0\n", "", 1) + text[j:]
        return text

    def _fs_iterative(self, text):
        """the free-surface solvers (Zb, Zs) with BiCGStab + ILU1 instead of the direct solver"""
        a = ('  Linear System Solver = Direct\n'
             '  Linear System Direct Method = $directsolver\n'
             '  Linear System Abort Not Converged = True\n'
             '  Linear System Convergence Tolerance = 1.0e-12')
        b = ('  Linear System Solver = Iterative\n'
             '  Linear System Iterative Method = BiCGStab\n'
             '  Linear System Preconditioning = ILU1\n'
             '  Linear System Max Iterations = 2000\n'
             '  Linear System Abort Not Converged = True\n'
             '  Linear System Convergence Tolerance = 1.0e-12')
        for name in ('Equation = "Free Surface Sea/Shelf"', 'Equation = "Free Surface Top"'):
            k = text.find(name)
            if k < 0:
                continue
            j = text.index("\nEnd", k)
            block = text[k:j]
            if a not in block:
                raise ValueError(f"linear_free_surface: no direct-solver lines in {name}")
            text = text[:k] + block.replace(a, b) + text[j:]
        return text

    @staticmethod
    def _section_end(text, marker):
        """index of the "\nEnd" closing the section that contains marker"""
        k = text.index(marker)
        return text.index("\nEnd", k)

    def _insert_in(self, text, marker, lines):
        """lines (str) added at the end of the section containing marker"""
        j = self._section_end(text, marker)
        return text[:j] + "\n" + lines.rstrip("\n") + text[j:]

    @staticmethod
    def _add_active(text, eq_name, ids):
        """append solver numbers to the Active Solvers list of the equation named eq_name"""
        k = text.index(f'Name = "{eq_name}"')
        m = re.compile(r"Active Solvers\s*\((\d+)\)\s*=\s*([\d ]+)").search(text, k)
        old = m.group(2).split()
        new = old + [str(i) for i in ids]
        return text[:m.start()] + f"Active Solvers({len(new)}) = {' '.join(new)}" + text[m.end():]

    def _sea_level_param(self, text):
        """sea level = the Lua parameter ess_sl in the constants and in every material (see sea_level_param)"""
        a = "  Sea Level = Real 0.0"
        if text.count(a) != 2:
            raise ValueError("sea_level_param: the sea level lines of the template changed")
        text = text.replace(a, "  Sea Level = Real #ess_sl")
        for m in ("Material 1", "Material 2"):              # the parent material of a bed element may be either
            k = text.index("\n" + m + "\n") + len(m) + 2
            text = text[:k] + "  Sea Level = Real #ess_sl\n" + text[k:]
        return text

    def _stokes_iterations(self, text, n):
        """the Stokes solver: exactly n nonlinear iterations (contact grounding line, see contact_stokes_iters)"""
        k = text.index('Equation = "Stokes-Vec"')
        i = text.rindex("\nSolver ", 0, k)
        j = text.index("\nEnd", k)
        sec = text[i:j]
        a = "  Nonlinear System Min Iterations = 1\n  Nonlinear System Max Iterations = 1"
        if a not in sec:
            raise ValueError("contact_stokes_iterations: the Stokes nonlinear iteration lines of the template changed")
        sec = sec.replace(a, f"  ! contact grounding line: a second iteration applies the nodes SlidCoef_Contact ungrounded\n"
                             f"  Nonlinear System Min Iterations = {n}\n  Nonlinear System Max Iterations = {n}")
        return text[:i] + sec + text[j:]

    def _shelf_cut(self, text):
        """EXPERIMENTAL (see shelf_cut_km): the ice elements with a node beyond x = shelf_cut_x are passive for the
        Stokes solver (Elmer fixes their velocities at the current values). With shelf_cut_target, the face between
        active and passive elements (Elmer's "Passive Target" boundary) carries the normal stress of a free-floating,
        laterally uniform shelf of the local thickness H: sigma_nn = -rho_i g depth + rho_i g H (1 - rho_i/rho_w) / 2
        (cryostatic plus the uniform tension that balances the sea water on the shelf; Elmer/Ice MISMIP_FS-SSA puts
        the cryostatic part plus the SSA loads there). Without it, the face is stress free (only to test Elmer's
        passive machinery)."""
        if not self.marine:
            raise ValueError("shelf_cut_km: marine template only")
        k = text.index('Name = "bodyforce ice"')
        j = text.index("\nEnd", k)
        text = (text[:j] + "\n  ! shelf cut (shelf_cut_km): the Stokes problem ends at the cut, the shelf beyond is passive"
                f"\n  Passive Element Min Nodes = Integer 1\n  Flow Solution Passive = Variable Coordinate 1\n"
                f'    Real Lua "tx[0] - {self.shelf_cut_x!r}"' + text[j:])
        if not self.shelf_cut_target:
            return text
        n = max(int(m) for m in re.findall(r"^Boundary Condition (\d+)\s*$", text, re.M))
        bc = (f"\nBoundary Condition {n + 1}\n"
              '  Name = "shelf cut"\n'
              "  ! the active/passive interface of the shelf cut: the normal stress of a free-floating shelf\n"
              "  Passive Target = Logical True\n"
              "  Flow Force BC = Logical True\n"
              "  External Pressure = Variable depth, height\n"
              '    Real Lua "-rhoi*gravity*tx[0] + 0.5*rhoi*gravity*(tx[0] + tx[1])*(1 - rhoi/rhow)"\n'
              "  ComputeNormal = Logical False\n"
              "End\n")
        m = list(re.finditer(r"^Boundary Condition \d+\s*$", text, re.M))[-1]
        e = text.index("\nEnd", m.start()) + len("\nEnd")
        return text[:e] + "\n" + bc + text[e:]

    def _fs_ssa(self, text):
        """the moving full-Stokes region (see fs_ssa_km): the grounding-line distance solver, the passive condition
        on the ice and the active/passive interface with the free-floating shelf stress"""
        if not self.marine:
            raise ValueError("fs_ssa_km: marine template only")
        if self.shelf_cut_x is not None:
            raise ValueError("fs_ssa_km and shelf_cut_km exclude each other")
        lib = os.path.join(os.path.dirname(self.ess_lib), "GLDistance")
        n = max(int(v) for v in re.findall(r"^Solver (\d+)", text, re.M)) + 1
        sol = (f"Solver {n}\n"
               '  Exec Solver = "Before Timestep"\n'
               '  Equation = "GL distance"\n'
               f'  Procedure = File "{lib}" "GLDistance"\n'
               '  Variable = -nooutput "GL Distance"\n'
               "  Variable DOFs = 1\n"
               "  No Matrix = Logical True\n"
               "End\n")
        ssa = self.fs_ssa_shelf == "ssa"
        if ssa:     # the SSA shelf (fs_ssa_shelf): right after the distance, before every time step, on the ice base
            sol += (f"\nSolver {n + 1}\n"
                    '  Exec Solver = "Before Timestep"\n'
                    '  Equation = "FS SSA shelf"\n'
                    f'  Procedure = File "{lib}" "FSSSAShelf"\n'
                    '  Variable = -dofs 2 "SSA Velocity"\n'
                    '  Exported Variable 1 = -dofs 4 "SSA Strain"\n'
                    f"  FS SSA Distance = Real {self.fs_ssa_x!r}\n"
                    "  FS SSA Viscosity = Real #viscosity\n"
                    f"  FS SSA Walls = Logical {not self.periodic_sides}\n"
                    "  Linear System Solver = Direct\n"
                    "  Linear System Direct Method = $directsolver\n"
                    "  Linear System Abort Not Converged = Logical True\n"
                    "  Linear System Convergence Tolerance = 1.0e-12\n"
                    "  Nonlinear System Max Iterations = 1\n"
                    "  Steady State Convergence Tolerance = 1.0e-5\n"
                    "End\n")
        k = text.index("!---------------- BOUNDARY CONDITIONS")
        k = text.rindex("\n", 0, k - 1)
        text = (text[:k] + "\n! the horizontal distance to the grounding line, every time step ([elmer] fs_ssa_km)\n"
                + sol + text[k:])
        text = self._add_active(text, "ice equations", [n])
        if ssa:
            text = self._add_active(text, "interface equations", [n + 1])
        # passive beyond the region: the ice, and the boundary elements of its base and surface (bodies of their own:
        # otherwise the sea pressure and spring of the passive shelf's base would still act on the interface nodes,
        # as Elmer/Ice's MISMIP_FS-SSA test also makes its base body passive)
        for bf in ("bodyforce ice", "bodyforce interface", "bodyforce freesurface"):
            k = text.index(f'Name = "{bf}"')
            j = text.index("\nEnd", k)
            text = (text[:j] + "\n  ! the moving full-Stokes region ([elmer] fs_ssa_km): passive beyond it for Stokes"
                    "\n  Passive Element Min Nodes = Integer 1\n  Flow Solution Passive = Variable GL Distance\n"
                    f'    Real Lua "tx[0] - {self.fs_ssa_x!r}"' + text[j:])
        # the shelf's force on the interface: "loads" (default: added to the Stokes right-hand side by
        # GLDistance's FSSSALinsolve), "elmer" (Elmer's Passive Target boundary: crashes in 3D, see FSSSALinsolve),
        # "0" (none: the interface stress free; tests). Env ESS_FS_SSA_TARGET.
        target = os.environ.get("ESS_FS_SSA_TARGET", "loads")
        if target == "0":
            return text
        if target == "loads":
            i0 = text.index('"IncompressibleNSVec"')
            i1 = text.index("\nEnd", i0)
            sec = text[i0:i1]
            rot = "RotateInitialGuess" in sec
            sec = re.sub(r"\n *Before Linsolve = .*", "", sec)
            sec += (f'\n  Before Linsolve = File "{lib}" "FSSSALinsolve"   ! [elmer] fs_ssa_km: the shelf force on the interface'
                    f"\n  FS SSA Distance = Real {self.fs_ssa_x!r}"
                    + ("\n  FS SSA Rotate Guess = Logical True" if rot else ""))
            if ssa:
                if re.search(r"\n *After Linsolve *=", sec):
                    raise ValueError("fs_ssa_shelf = ssa: the Stokes solver already has an After Linsolve procedure")
                sec += ("\n  FS SSA Shelf Stress = Logical True   ! the interface stress from the SSA shelf's strain rate"
                        "\n  FS SSA Viscosity = Real #viscosity"
                        f'\n  After Linsolve = File "{lib}" "FSSSAAfterLinsolve"   ! the passive shelf\'s plug flow back')
            return text[:i0] + sec + text[i1:]
        if target != "elmer":
            raise ValueError(f"ESS_FS_SSA_TARGET must be loads, elmer or 0, not {target!r}")
        nb = max(int(m) for m in re.findall(r"^Boundary Condition (\d+)\s*$", text, re.M))
        bc = (f"\nBoundary Condition {nb + 1}\n"
              '  Name = "full-Stokes / shelf interface"\n'
              "  ! the active/passive interface ([elmer] fs_ssa_km): the normal stress of a free-floating shelf\n"
              "  Passive Target = Logical True\n"
              "  Flow Force BC = Logical True\n"
              "  External Pressure = Variable depth, height\n"
              '    Real Lua "-rhoi*gravity*tx[0] + 0.5*rhoi*gravity*(tx[0] + tx[1])*(1 - rhoi/rhow)"\n'
              "  ComputeNormal = Logical False\n"
              "End\n")
        m = list(re.finditer(r"^Boundary Condition \d+\s*$", text, re.M))[-1]
        e = text.index("\nEnd", m.start()) + len("\nEnd")
        return text[:e] + "\n" + bc + text[e:]

    def _surface_dirichlet(self, text):
        """the upper surface: T = the surface temperature (Tsurf, or the warm snout ramp) instead of the transfer
        condition"""
        a = (f'  Heat Flux BC = Logical True\n  Heat transfer coefficient = Real {self.surface_transfer:g}\n')
        k = text.find('Name = "free surface"')
        if k < 0 or a not in text[k:k + 1500]:
            raise ValueError("surface_bc = dirichlet: the surface heat condition of the template changed")
        i = text.index(a, k)
        j = text.index("External Temperature =", i)
        return text[:i] + text[i + len(a):j] + "Temperature =" + text[j + len("External Temperature ="):]

    def _sliding_temperate(self, text):
        """Weertman friction as at the melting point everywhere: TDepSliding(fcoeff, T, dT, x) -> fcoeff"""
        a = "TDepSliding(fcoeff, tx[0], dT, tx[1]) * ess_grounded(tx[2])"
        if a not in text:
            raise ValueError("sliding = temperate: the Weertman Friction Coefficient expression of the template changed")
        text = text.replace(a, "fcoeff * ess_grounded(tx[2])")
        # flux friction heating: the same coefficient (C = fcoeff, the y bump, grounded) times |u_b|^2
        f = ('  Heat Flux = Variable Temperature, Velocity 1, Velocity 2, Velocity 3, Coordinate 1\n'
             '    Real Lua "TDepSliding(fcoeff, tx[0], dT, tx[4]) * (tx[1]*tx[1] + tx[2]*tx[2] + tx[3]*tx[3])"')
        if f in text:
            bump = (f" * (1 + {self.slide_ybump!r} * math.sin(2 * math.pi * tx[3] / {self.width!r}))"
                    if self.slide_ybump else "")
            text = text.replace(f, '  Heat Flux = Variable Velocity 1, Velocity 2, Velocity 3, Coordinate 2, GroundedMask\n'
                                   f'    Real Lua "fcoeff * ess_grounded(tx[4]){bump} * (tx[0]*tx[0] + tx[1]*tx[1] + tx[2]*tx[2])"')
        return text

    def _thermal_loads(self, text):
        """friction heating from the Stokes loads, periodic-aware (see [elmer] thermal = loads). With periodic sides
        a periodic copy of the loads ("Periodic Flow Solution Loads", FixPeriodicLoads folded) is always made (run
        by the heat solver first, so it holds this step's loads); periodic_gl = fix: the heat load and the Friction
        Loads export use it; elmerstreams: the raw loads halved at y < 1; none: the raw loads."""
        if self.stokes_only:
            return text
        if not self.marine or self.gl_mode != "contact":
            raise ValueError("[elmer] thermal = 'loads': marine template with the contact grounding line only")
        old = ('Variable Flow Solution Loads 1, Flow Solution Loads 2, '
               'Flow Solution Loads 3, Velocity 1, Velocity 2, Velocity 3, Coordinate 2\n'
               '    Real Lua "frictionloads(tx[0],tx[1],tx[2],tx[3],tx[4],tx[5], W, W)"')
        n_old = 2 if self.friction_heat == "loads" else 1           # the heat load (loads only) and the export
        if text.count(old) != n_old:
            raise ValueError("thermal = loads: the loads friction-heating lines of the template changed")
        if self.periodic_gl == "elmerstreams":
            text = text.replace(old, old.replace("tx[5], W, W)", "tx[5], tx[6], W)"))
        elif self.periodic_gl == "fix":
            text = text.replace(old, old.replace("Flow Solution Loads ", "Periodic Flow Solution Loads "))
        if not self.periodic_sides:
            return text
        n = max(int(v) for v in re.findall(r"^Solver (\d+)", text, re.M))
        a, b = n + 1, n + 2
        lib = self.periodic_lib
        fpl = lambda num, ex, eq: (f'Solver {num}\n  Exec Solver = "{ex}"\n  Equation = "{eq}"\n'
                               f'  Procedure = "{lib}" "FixPeriodicLoads"\n  Source Variable = String "Flow Solution Loads"\n'
                               f'  Target Variable = String "Periodic Flow Solution Loads"\n'
                               f'  Periodic Mode = String "folded"\nEnd\n')
        k = text.index("!---------------- BOUNDARY CONDITIONS")
        k = text.rindex("\n", 0, k - 1)
        text = (text[:k] + "\n! periodic copy of the Stokes loads (friction heating): created before all, then run by "
                "the heat solver first (Pre Solvers)\n" + fpl(a, "Before All", "Periodic Flow Solution Loads Init") + "\n" + fpl(b, "Never", "Periodic Flow Solution Loads") + text[k:])
        text = self._add_active(text, "ice equations", [a, b])
        return self._insert_in(text, 'Equation = String "Homologous Temperature Equation"', f"  Pre Solvers(1) = {b}")

    def _thermal_elmerstreams(self, text):
        """ElmerStreams Marine_Term's thermal chain (see [elmer] thermal)"""
        if not self.marine or self.gl_mode != "contact":
            raise ValueError("[elmer] thermal = 'elmerstreams': marine template with the contact grounding line only")
        if self.friction_heat != "loads":
            raise ValueError("[elmer] thermal = 'elmerstreams' needs friction_heat = 'loads' (as ElmerStreams)")
        if not self.strain_heat or self.stokes_only:
            raise ValueError("[elmer] thermal = 'elmerstreams' needs strain_heat = true and the heat solve")
        fix, halve = self.periodic_gl == "fix", self.periodic_gl == "elmerstreams"
        n = max(int(v) for v in re.findall(r"^Solver (\d+)", text, re.M))
        ids = iter(range(n + 1, n + 20))
        # numbers in execution order within a step: the periodic copies of the heat loads before the sheet solver
        ct, stress = next(ids), next(ids)
        fixT = fixW = fixU0 = fixU = None
        if fix:
            fixT, fixW = next(ids), next(ids)
        sheet = next(ids)
        if fix:
            fixU0, fixU = next(ids), next(ids)
        it = ('  Linear System Solver = "Iterative"\n  Linear System Iterative Method = "BiCGStab"\n'
              '  Linear System Max Iterations = 300\n  Linear System Convergence Tolerance = 1.0E-09\n'
              '  Linear System Abort Not Converged = False\n  Linear System Preconditioning = "ILU0"\n'
              '  Linear System Residual Output = 100\n')
        lib = self.periodic_lib
        blocks = [f"""
! ---- ElmerStreams thermal chain ([elmer] thermal = "elmerstreams")
Solver {ct}
  Exec Solver = "Before Timestep"
  Equation = "ColdTempMask"
  Procedure = "{self.thermal_libs['coldtemp']}" "ColdTempMaskSolver"
  Variable = "ColdTempMask"
  Variable DOFs = 1
  Toler = Real 1e-3
End

Solver {stress}
  Equation = String "StressSolver"
  Exec Solver = "after timestep"
  Procedure = File "ElmerIceSolvers" "ComputeDevStress"
  Variable = -nooutput "Sij"
  Variable DOFs = 1
  Flow Solver Name = String "Flow Solution"
  Stress Variable Name = String "Stress"
  Exported Variable 1 = "Stress"
  Exported Variable 1 DOFs = 6
{it}End
"""]
        if fix:
            fpl = lambda num, ex, src, tgt, mode, sfx="": (
                f'Solver {num}\n  Exec Solver = "{ex}"\n  Equation = "Periodic {src}{sfx}"\n'
                f'  Procedure = "{lib}" "FixPeriodicLoads"\n  Source Variable = String "{src}"\n'
                f'  Target Variable = String "{tgt}"\n  Periodic Mode = String "{mode}"\nEnd\n')
            blocks.append("\n! periodic copies of the loads (FixPeriodicLoads): the melt (after the heat solve, before the "
                          "sheet)\n" + fpl(fixT, "After Timestep", "Temperature Loads", "Periodic Temperature Loads", "folded")
                          + "\n" + fpl(fixW, "After Timestep", "Strain Heating Boundary Weights",
                                        "Periodic Strain Heating Boundary Weights", "split")
                          + "\n! the friction heating: created before all, then run by the heat solver first (Pre Solvers)\n"
                          + fpl(fixU0, "Before All", "Flow Solution Loads", "Periodic Flow Solution Loads", "folded", " Init")
                          + "\n" + fpl(fixU, "Never", "Flow Solution Loads", "Periodic Flow Solution Loads", "folded"))
        blocks.append(f"""
Solver {sheet}
  Exec Solver = "After Timestep"
  Equation = String "Sheet solver water sheet thickness"
  Variable = String "Water Sheet Thickness"
  Procedure = "{self.thermal_libs['sheet']}" "SheetSolverhw"
  Linear System Solver = "Iterative"
  Linear System Iterative Method = "GCR"
  Linear System GCR Restart = Integer 200
  Linear System Max Iterations = 1500
  Linear System Convergence Tolerance = 1.0e-8
  Linear System Abort Not Converged = False
  Linear System Preconditioning = "ILU8"
  Linear System Residual Output = 100
  Nonlinear System Convergence Tolerance = 1.0e-6
  Nonlinear System Max Iterations = 200
  Nonlinear System Min Iterations = 1
  Nonlinear System Relaxation Factor = Real 0.5
  Nonlinear System Convergence Measure = "solution"
  Nonlinear System Reset Newton = Logical True
  Nonlinear System Newton After Iterations = 20
  Nonlinear System Newton After Tolerance = 1.0e-2
  Steady State Convergence Tolerance = 1.0E-05
  Exported Variable 1 = -dofs 1 "Melt rate"
  Exported Variable 2 = -dofs 1 "Effective Pressure Output"
  Exported Variable 3 = -dofs 3 "Water Flux"
  Exported Variable 4 = -dofs 1 "Hydraulic Conductivity Output"
  Nonlinear Update Exported Variables = Logical True
  Update Exported Variables = Logical True
End
""")
        k = text.index("!---------------- BOUNDARY CONDITIONS")
        k = text.rindex("\n", 0, k - 1)
        text = text[:k] + "\n" + "".join(blocks) + text[k:]
        # the mask now belongs to its solver: not exported by the frozen-field owner any more
        j = self._section_end(text, '"ESSNoOp"')
        i = text.rindex("Solver ", 0, j)
        text = text[:i] + text[i:j].replace('  Exported Variable 1 = -dofs 1 "ColdTempMask"\n', "")\
            .replace('  Exported Variable 1 = -dofs 1 "ColdTempMask"', "") + text[j:]
        text = self._add_active(text, "ice equations", [stress] + ([fixT, fixW, fixU0, fixU] if fix else []))
        text = self._add_active(text, "interface equations", [ct, sheet])
        # heat solver: the nodal coordinates (sheet solver), and the periodic friction loads first
        heat = 'Equation = String "Homologous Temperature Equation"'
        text = self._insert_in(text, heat, '  Exported Variable 4 = -dofs 3 "NodalCoords"'
                               + (f"\n  Pre Solvers(1) = {fixU}" if fix else ""))
        for bf in ("bodyforce bedrock", "bodyforce filler", "bodyforce ice"):
            text = self._insert_in(text, f'Name = "{bf}"', "\n".join(f"  NodalCoords {c} = Equals Coordinate {c}"
                                                                     for c in (1, 2, 3)))
        loads = "Periodic Temperature Loads" if fix else "Temperature Loads"
        weights = "Periodic Strain Heating Boundary Weights" if fix else "Strain Heating Boundary Weights"
        yarg = "tx[5]" if halve else "W"
        melt = (f'Variable "{loads}", "{weights}", "Friction Loads", "GroundedMask", "Time", "Coordinate 2"\n'
                f'    Real LUA "getmeltrate(tx[0],tx[1],tx[2],tx[3],tx[4],{yarg}, W) + Source*tx[2]"')
        text = self._insert_in(text, 'Name = "bodyforce interface"', f"""  ! water sheet (ElmerStreams)
  Melt Source = {melt}
  Melt Rate = {melt}
  Effective Pressure Output = Variable "Water Sheet Thickness", "Stress 3"
    Real Lua "N(tx[0], tx[1], N0, hr)"
  Hydraulic Conductivity Output = Variable "Effective Pressure Output"
    Real LUA "conductivity(tx[0])"
  Basal Velocity 2 = Equals Velocity 2
  Basal Velocity 3 = Equals Velocity 3""")
        text = self._insert_in(text, 'Name = "ice material"', """  ! Hydrology model (ElmerStreams)
  Density Water = Real #rhow
  Diffusion coefficient = Real #k
  Bedrock bump height = Real #hr
  Cavity spacing = Real #lr
  Phi0 = Variable "Stress 3", "zb"
    Real LUA "-tx[0] + rhow*gravity*tx[1]"
  Effective Pressure = Variable "Water Sheet Thickness", "Stress 3"
    Real Lua "N(tx[0], tx[1], N0, hr)"
  Hydraulic Conductivity = Variable "Effective Pressure"
    Real LUA "conductivity(tx[0])"
  Effective Pressure First Derivative = Variable "Water Sheet Thickness", "Stress 3"
    Real LUA "dNdh(tx[0],tx[1], N0, hr)"
  Effective Pressure Second Derivative = Variable "Water Sheet Thickness"
     Real LUA "2*N0*(hr + eps)/((tx[0] + eps)^3)"
  Hydraulic Conductivity First Derivative = Variable "Effective Pressure"
    Real LUA "dkappadN(tx[0])"\n""")
        text = self._insert_in(text, "Initial Condition 4", '  Water Sheet Thickness = Variable "Coordinate 1"\n'
                               '    Real Lua "InitialSheet(tx[0])"')
        # bed: the sheet at hr under floating ice; friction heating from the (periodic) loads
        text = self._insert_in(text, 'Name = "interface"\n  Body ID = 4',
                               '  Water Sheet Thickness = Real #hr\n'
                               '  Water Sheet Thickness Condition = Variable "Time","GroundedMask"\n'
                               '    Real LUA "IfThenElse(tx[0]>1E-5, -tx[1], -1)"')
        old = ('  Temperature Load = Variable Flow Solution Loads 1, Flow Solution Loads 2, '
               'Flow Solution Loads 3, Velocity 1, Velocity 2, Velocity 3, Coordinate 2\n'
               '    Real Lua "frictionloads(tx[0],tx[1],tx[2],tx[3],tx[4],tx[5], W, W)"')
        if old not in text:
            raise ValueError("thermal: the loads friction-heating lines of the template changed")
        new = old
        if fix:
            new = old.replace("Flow Solution Loads ", "Periodic Flow Solution Loads ")
        elif halve:
            new = old.replace("tx[5], W, W)", "tx[5], tx[6], W)")
        text = text.replace(old, new)
        # Lua: the hydrology functions and parameters after parameters.lua
        k = text.index("Header")
        text = text[:k] + f"!---LUA BEGIN\n! assert(loadfile('{self.hydro_lua}'))()\n!---LUA END\n\n" + text[k:]
        return text

    def _slide_ybump(self, text):
        """the sliding coefficient times 1 + slide_ybump sin(2 pi y / width)"""
        a = ('  Weertman Friction Coefficient = Variable Temperature, Coordinate 1, GroundedMask\n'
             '    Real lua "TDepSliding(fcoeff, tx[0], dT, tx[1]) * ess_grounded(tx[2])"')
        if a not in text:
            raise ValueError("slide_ybump: the Weertman Friction Coefficient lines of the template changed")
        b = ('  Weertman Friction Coefficient = Variable Temperature, Coordinate 1, GroundedMask, Coordinate 2\n'
             f'    Real lua "TDepSliding(fcoeff, tx[0], dT, tx[1]) * ess_grounded(tx[2]) * (1 + {self.slide_ybump!r} '
             f'* math.sin(2 * math.pi * tx[3] / {self.width!r}))"')
        return text.replace(a, b)

    def _cartesian_bed(self, text):
        """the bed boundary condition ("interface") in x/y/z components (see cartesian_bed)"""
        if self.marine and self.gl_mode == "contact":
            raise ValueError("cartesian_bed: the contact grounding line needs normal-tangential velocities; use the "
                             "frozen one (ess run --gl frozen)")
        k = text.index("Normal-Tangential Velocity = Logical True")      # the bed condition (the only N-T one)
        i = text.rindex("Boundary Condition", 0, k)
        j = text.index("\nEnd", k)
        bc = text[i:j]
        swaps = [("  Normal-Tangential Velocity = Logical True", "  Normal-Tangential Velocity = Logical False"),
                 ("  Velocity 1 = 0.0", "  Velocity 3 = 0.0"),                       # no flow through the bed
                 ("  Velocity 1 Condition", "  Velocity 3 Condition")]
        for a, b in swaps:
            if a in bc:
                bc = bc.replace(a, b)
        # slip coefficients: N-T (1 normal, 2/3 tangential) -> x/y/z (3 vertical, 1/2 horizontal), via markers
        bc = (bc.replace("  Slip Coefficient 1 =", "  @S3@ =").replace("  Slip Coefficient 2 =", "  @S1@ =")
                .replace("  Slip Coefficient 3 =", "  @S2@ ="))
        for k in "123":
            bc = bc.replace(f"  @S{k}@ =", f"  Slip Coefficient {k} =")
        if "Normal-Tangential Velocity = Logical False" not in bc:
            raise ValueError("cartesian_bed: no Normal-Tangential Velocity line in the interface condition")
        return text[:i] + bc + text[j:]

    def state_records(self, x) -> Records:
        rec = dict(self.layout.to_records(np.asarray(x, float)))
        rec.update(self.frozen)
        rec.update(self.warm_records)
        return rec

    # ---------------------------------------------------------------- running Elmer
    def _launch(self, run_dir, tag):
        cmd = shlex.split(self.mpirun.format(n=self.n_procs)) + [self.solver, "run.sif"]
        logf = os.path.join(run_dir, "elmer.log")
        t0 = time.time()
        with open(logf, "w") as f:
            try:
                ret = subprocess.run(cmd, cwd=run_dir, stdout=f, stderr=subprocess.STDOUT,
                                     timeout=self.timeout).returncode
            except subprocess.TimeoutExpired:
                raise ModelFailure(f"{tag}: Elmer timed out after {self.timeout:.0f} s (log {logf})")
            except FileNotFoundError as e:
                raise ModelFailure(f"{tag}: cannot launch {cmd[0]}: {e}")
        dt = time.time() - t0
        with open(logf, errors="replace") as f:
            log = f.read()
        tail = log[-20000:]
        # the whole log: per-rank warnings at the end (e.g. StateIO at 768 ranks) can push ALL DONE out of the tail
        if ret != 0 or not ("ALL DONE" in log or "ELMER SOLVER FINISHED" in tail):
            err = [l for l in tail.splitlines() if "ERROR" in l or "Fatal" in l or "FATAL" in l][-3:]
            raise ModelFailure(f"{tag}: Elmer failed (exit {ret}) after {dt:.0f} s; see {logf}"
                               + ("\n  " + "\n  ".join(err) if err else ""))
        return dt

    def _read_out(self, run_dir, base, tag) -> Records:
        files = partition_files(os.path.join(run_dir, base))
        if len(files) != self.n_procs:
            raise ModelFailure(f"{tag}: expected {self.n_procs} output files {base}.*, found {len(files)}")
        rec, mism = merge_partitions(files)
        bad = {k: v for k, v in mism.items() if v > 1e-8}
        if bad:
            log.warning(f"{tag}: shared-node values differ between partitions: " +
                        ", ".join(f"{k} {v:.1e}" for k, v in bad.items()))
        return rec

    def _key(self, x, p, spec):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(x, "<f8").tobytes())
        h.update(repr((float(p), spec.dt, spec.n_steps, spec.coupled_iters, spec.rock_heat_capacity, self.n_procs)).encode())
        h.update(self.build_sif(spec).encode())
        h.update(self.params_lua(p).encode())
        with open(self.ess_utils, "rb") as f:     # its functions (SMB, sliding, ...) shape the map
            h.update(f.read())
        for k in sorted(self.frozen):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.frozen[k][1], "<f8").tobytes())
        for k in sorted(self.warm_records):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.warm_records[k][1], "<f8").tobytes())
        h.update(self._mesh_stamp().encode())
        return h.hexdigest()[:40]

    def _mesh_stamp(self) -> str:
        """size and mtime of the mesh files: a re-meshed case (`ess setup`) never reuses cached runs"""
        if getattr(self, "_mesh_stamp_v", None) is None:
            mdir = os.path.join(self.case_dir, self.mesh_db)
            items = []
            for root, _, files in os.walk(mdir):
                for f in sorted(files):
                    st = os.stat(os.path.join(root, f))
                    items.append(f"{os.path.relpath(os.path.join(root, f), mdir)}:{st.st_size}:{int(st.st_mtime)}")
            self._mesh_stamp_v = ";".join(sorted(items))
        return self._mesh_stamp_v

    def _new_run_dir(self, tag):
        self.nrun += 1
        d = os.path.join(self.work, "runs", f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}_{uuid.uuid4().hex[:6]}")
        os.makedirs(d)
        return d

    def run(self, x, p, spec: MapSpec, tag="run", want_diag=False):
        if self.layout is None:
            raise RuntimeError("ElmerModel.run needs a state layout (run `ess extract` first)")
        key = self._key(x, p, spec)
        cache = os.path.join(self.work, "cache", key + ".npz")
        if self.use_cache and os.path.exists(cache):
            with np.load(cache) as z:
                xo = z["x"].copy()
                diag = {k[3:]: (z["g::" + k[3:]], z[k]) for k in z.files if k.startswith("v::")} if want_diag else {}
            log.debug(f"{tag}: cache hit {key[:10]}")
            return xo, diag

        run_dir = self._new_run_dir(tag)
        with open(os.path.join(run_dir, "parameters.lua"), "w") as f:
            f.write(self.params_lua(p))
        write_state_file(os.path.join(run_dir, "state_in.bin"), self.state_records(x))
        with open(os.path.join(run_dir, "run.sif"), "w") as f:
            f.write(self.build_sif(spec, runname=tag))
        secs = self._launch(run_dir, tag)
        rec = self._read_out(run_dir, "state_out.bin", tag)
        xo = self.layout.to_vector(rec)
        names = {find_name(rec, n) for n in self.state_vars}
        diag = {k: v for k, v in rec.items() if k not in names}
        log.debug(f"{tag}: Elmer {secs:.0f} s in {run_dir}")
        if self.use_cache:
            d = {"x": xo}
            for k, (g, v) in diag.items():
                d["g::" + k] = g
                d["v::" + k] = v
            np.savez(cache + ".tmp.npz", **d)
            os.replace(cache + ".tmp.npz", cache)
        if not self.keep_runs:
            shutil.rmtree(run_dir, ignore_errors=True)
        return xo, (diag if want_diag else {})

    # ---------------------------------------------------------------- first state
    def extract(self, restart_file: str | None = None, restart_before_ics: bool = False, p=None):
        """Run the template once (from the initial conditions of `ess setup`, or from a restart file)
        and write the state as it is at the start.
        Returns (records, layout, frozen)."""
        p = self.cfg["parameter"]["value"] if p is None else p
        run_dir = self._new_run_dir("extract")
        with open(os.path.join(run_dir, "parameters.lua"), "w") as f:
            f.write(self.params_lua(p))
        names0 = list(dict.fromkeys(self.state_vars + self.frozen_vars))
        sif = self.build_sif(MapSpec(1e-12, 1), state_in="none", state_out0="state0.bin", write0=names0,
                             restart=restart_file, restart_before_ics=restart_before_ics, runname="extract")
        with open(os.path.join(run_dir, "run.sif"), "w") as f:
            f.write(sif)
        self._launch(run_dir, "extract")
        rec = self._read_out(run_dir, "state0.bin", "extract")
        scales = dict(self.cfg["state"].get("scales", {}))
        layout = StateLayout.from_records(rec, self.state_vars, scales)
        frozen = {n: rec[find_name(rec, n)] for n in self.frozen_vars}
        log.info(f"extract: state of {layout.size} values (" +
                 ", ".join(f"{b.name} {len(b.gids)}" for b in layout.blocks) + "); frozen: " +
                 ", ".join(f"{k} {len(v[0])}" for k, v in frozen.items()))
        if not self.keep_runs:
            shutil.rmtree(run_dir, ignore_errors=True)
        return rec, layout, frozen


# -------------------------------------------------------------------- consistency of the frozen masks

def mask_report(layout: StateLayout, x, frozen: Records, diag: Records, T0=273.15, marine=False,
                rho_ratio=None) -> dict:
    """Where would the frozen masks be wrong at this state? (ess.masks.update_masks, nothing changed)
    cold_above_Tm:  cold-bed nodes with T > T0 (would switch to temperate)
    temp_freezing:  temperate nodes with basal refreezing (loads > 0), if loads are available
    marine: to_grounded / to_floating (grounding-line nodes that would switch), x_gl (m)"""
    from .masks import update_masks
    try:
        _, _, rep = update_masks(layout, x, frozen, diag, marine=marine, rho_ratio=rho_ratio, apply=False)
    except KeyError as e:
        log.debug(f"mask report: {e}")
        return {}
    rep["cold_above_Tm"] = rep.pop("to_temperate", 0)
    rep["temp_freezing"] = rep.pop("to_cold", 0)
    return rep
