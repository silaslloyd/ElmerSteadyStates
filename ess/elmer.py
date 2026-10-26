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
        self.mesh_db = e.get("mesh_db", "footprint")
        self.ess_lib = cfg["_path"](e.get("ess_lib", "elmer/SteadyStateIO"))
        self.ess_utils = cfg["_path"](e.get("ess_utils", "elmer/ess_utils.lua"))
        self.params_file = cfg["_path"](cfg.get("case", {}).get("parameters", "case/parameters.lua"))
        # ESS_NPROCS / ESS_MPIRUN (environment) override [elmer] n_procs / mpirun: several drivers in one batch
        # job, each with its own share of the cores (jobs/stokes_tests_*.cmd)
        self.n_procs = (int(os.environ.get("ESS_NPROCS", 0)) or int(e.get("n_procs", 0))
                        or int(os.environ.get("SLURM_NTASKS", "1")))
        self.mpirun = os.environ.get("ESS_MPIRUN") or e.get("mpirun", "mpirun -np {n}")
        self.solver = e.get("solver", "ElmerSolver")
        self.timeout = float(e.get("timeout", 6 * 3600))
        self.keep_runs = bool(e.get("keep_runs", False))
        self.use_cache = bool(e.get("cache", True))
        self.diagnostics = list(e.get("diagnostics", []))
        self.extra_subs = dict(e.get("substitutions", {}))
        self.vtu = bool(e.get("vtu", False))
        self.friction_heat = str(e.get("friction_heat", "flux"))
        self.strain_heat = bool(e.get("strain_heat", False))
        # heat-only runs (`ess temperate-sweep --heat-only`): the free surface is not solved, so the geometry
        # stays as in the input state; Stokes still runs (same velocity every step on a frozen geometry
        # when nothing in the flow depends on temperature)
        self.heat_only = False
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
        self.iterative_tol = float(e.get("iterative_tol", 1e-10))
        self.stokes_precond = str(e.get("stokes_preconditioner", "ILU2"))
        self.ilut_tol = float(e.get("ilut_tolerance", 1e-4))     # drop tolerance for ILUT
        # "P2/P1" (default) or "Stabilized" (P1/P1 with stabilisation: far easier for ILU-preconditioned
        # Krylov solvers, but a different discretisation, so slightly different steady states)
        self.stokes_element = str(e.get("stokes_element", "P2/P1"))
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
        self._check_inputs()

    # ---------------------------------------------------------------- set-up checks
    def _check_inputs(self):
        mesh = os.path.join(self.case_dir, self.mesh_db)
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

    def _linear_block(self, kind, which) -> str:
        """Linear-solver keywords for one solver. Iterative solves must converge far below the finite-
        difference perturbation (fd eps): their error is noise in the Jacobian products (check with
        `ess check`: the repeat difference rms(Phi_a - Phi_b) and the linearity plateau)."""
        if kind == "direct":
            return ('  Linear System Solver = Direct\n'
                    '  Linear System Direct Method = $directsolver\n'
                    '  Linear System Abort Not Converged = True\n'
                    '  Linear System Convergence Tolerance = 1.0e-12')
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
                    '  Linear System Abort Not Converged = True\n'
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
        return {"__EQ4_SOLVERS__": "(4) = 8 12 15 16",
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
""",
                "__GL_SLIDING__": weert.format(f="SlidCoef_Contact") + """
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
            "__MIN_ZS_EXPR__": self.min_zs_expr,
            "__LS_STOKES__": self._linear_block(self.linear_stokes, "stokes"),
            "__LS_HEAT__": self._linear_block(self.linear_heat, "heat"),
            "__LS_STRAIN__": self._linear_block(self.linear_heat, "strain"),
            "__STOKES_STAB__": f'"{self.stokes_element}"',
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
        left = sorted(set(PLACEHOLDER.findall(text)))
        if left:
            raise ValueError(f"unfilled placeholders in {self.template}: {', '.join(left)}")
        return text

    def state_records(self, x) -> Records:
        rec = dict(self.layout.to_records(np.asarray(x, float)))
        rec.update(self.frozen)
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
            tail = f.read()[-20000:]
        if ret != 0 or "ALL DONE" not in tail:
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
