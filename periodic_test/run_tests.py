#!/usr/bin/env python3
"""
Periodic-BC / partitioning test matrix for the thermo-mechanical ice slab (README.md).

Run inside a batch job with enough MPI tasks (jobs: periodic_test/run.cmd):
    python3 periodic_test/run_tests.py --np 8                 # every case
    python3 periodic_test/run_tests.py --np 8 --only A_ B_    # cases whose name contains A_ or B_
    python3 periodic_test/run_tests.py --list                 # print the cases
    python3 periodic_test/run_tests.py --analyse              # analysis only (finished cases)

Partitioned cases use N = --np partitions (default $SLURM_NTASKS, else 8; a multiple of 4).
Each case runs in periodic_test/out/np<N>/<case>/ (sif, params.lua, elmer.log, out.bin.<pe>);
the summary goes to periodic_test/out/np<N>/summary.csv and is printed.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
from make_mesh import write_footprint  # noqa: E402

# ---------------------------------------------------------------- geometry and physics (MPa - a - m)
GEOM = dict(L=30e3, W=8e3, H=600.0, nx=60, ny=16, nz_layers=10)
PARAMS = """
yearinsec = 31556926.0
rhoi = 917.0e-6 / yearinsec^2              -- MPa a2 m-2
gravity = 9.81 * yearinsec^2               -- m a-2
sina = 0.005                               -- surface slope driving the flow along x
cosa = math.sqrt(1 - sina^2)
eta = 3.17                                 -- MPa a (1e14 Pa s), Newtonian
Cslip = 1.5e-3                             -- MPa a m-1: tau_b = Cslip u_b (~30 m/a)
heatconductivity = 2.1 * yearinsec * 1.0e-6
heatcapacity = 2009.0 * yearinsec^2
G = 0.05 * yearinsec * 1.0e-6              -- geothermal flux (0.05 W m-2)
Ts = {Ts}                                  -- surface temperature (K)
Tb0 = {Tb0}                                -- initial basal temperature (K), linear profile to Ts
W = {W}
H = {H}
halve_y0 = {halve}                         -- ElmerStreams frictionloads: halve the load at y < 1 m
sliding_dT = {sdT}                         -- 0: linear sliding; > 0: Cslip exp(-min(T - T0, 0) / sliding_dT)

function max(a, b) if a > b then return a else return b end end

-- frictional heating as nodal loads (as ElmerStreams utils.lua frictionloads)
function frictionloads(lx, ly, lz, vx, vy, vz, y)
  local load = max(-lx * vx, 0) + max(-ly * vy, 0) + max(-lz * vz, 0)
  if halve_y0 and y < 1 then load = load / 2 end
  return load
end

-- frictional heating as a flux: tau_b . u_b = Cslip |u_b|^2 (+ geothermal)
-- basal drag coefficient: constant, or temperature dependent as ElmerStreams (capped at melting)
function slip(T)
  if sliding_dT <= 0 then return Cslip end
  local d = T - 273.15
  if d > 0 then d = 0 end
  return Cslip * math.exp(-d / sliding_dT)
end

function frictionflux(vx, vy, vz, T)
  return G + slip(T) * (vx * vx + vy * vy + vz * vz)
end
"""

# ---------------------------------------------------------------- partitionings of the footprint
# N partitions (--np, default $SLURM_NTASKS or 8); name -> (number of partitions, ElmerGrid options)
def parts_table(N):
    if N % 4:
        raise SystemExit(f"--np {N}: needs a multiple of 4 (the y-cuts are N/4 x 4 x 1, as stream.cmd)")
    return {
        "serial":    (1, []),
        "xcut":      (N, ["-partition", str(N), "1", "1"]),                     # cuts across x only
        "xcut_per":  (N, ["-partition", str(N), "1", "1", "-periodic", "0", "1", "0"]),
        "ycut":      (N, ["-partition", str(N // 4), "4", "1"]),                # as stream.cmd, no -periodic
        "ycut_per":  (N, ["-partition", str(N // 4), "4", "1", "-periodic", "0", "1", "0"]),   # as stream.cmd
        "metis":     (N, ["-metiskway", str(N)]),
        "metis_per": (N, ["-metiskway", str(N), "-periodic", "0", "1", "0"]),
    }


PARTS = parts_table(8)


# ---------------------------------------------------------------- linear solvers
LS = {
    "direct": """  Linear System Solver = Direct
  Linear System Direct Method = MUMPS""",
    # ElmerStreams heat solver
    "gcr_ilu2": """  Linear System Solver = "Iterative"
  Linear System Iterative Method = "GCR"
  Linear System GCR Restart = Integer 200
  Linear System Max Iterations = 15000
  Linear System Convergence Tolerance = 1.0E-08
  Linear System Abort Not Converged = True
  Linear System Preconditioning = "ILU2"
  Linear System Residual Output = 10""",
    "bicgstabl_ilu1": """  Linear System Solver = "Iterative"
  Linear System Iterative Method = "BiCGStabL"
  Linear System BiCGStabL Polynomial Degree = Integer 4
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0E-08
  Linear System Abort Not Converged = True
  Linear System Preconditioning = "ILU1"
  Linear System Residual Output = 10""",
    "gcr_none": """  Linear System Solver = "Iterative"
  Linear System Iterative Method = "GCR"
  Linear System GCR Restart = Integer 200
  Linear System Max Iterations = 15000
  Linear System Convergence Tolerance = 1.0E-08
  Linear System Abort Not Converged = True
  Linear System Preconditioning = "None"
  Linear System Residual Output = 10""",
    # ElmerStreams DeformationalHeat (strain heating) solver
    "gcr_ilu1": """  Linear System Solver = "Iterative"
  Linear System Iterative Method = "GCR"
  Linear System Max Iterations = 5000
  Linear System Convergence Tolerance = 1.0E-08
  Linear System Preconditioning = "ILU1"
  Linear System Residual Output = 100""",
    # ElmerStreams Stokes solver (linsys/gcr.sif): note Abort Not Converged = False
    "gcr_ilu8": """  Linear System Solver = Iterative
  Linear System Iterative Method = "gcr"
  Linear System GCR Restart = Integer 200
  Linear System Convergence Tolerance = 1.0E-8
  Linear System Max Iterations = 5000
  Linear System Abort Not Converged = False
  Linear System Preconditioning = "ILU8"
  Linear System Residual Output = 100""",
}

PERIODIC = """  Periodic BC = 3
  Periodic BC Translate(3) = Real 0.0 __W__ 0.0
  Periodic BC Velocity 1 = Logical True
  Periodic BC Velocity 2 = Logical True
  Periodic BC Velocity 3 = Logical True
  Periodic BC Pressure = Logical True
  Periodic BC Temperature = Logical True
__SH_PERIODIC__"""
WALL = "  Velocity 2 = Real 0.0        ! free slip; zero heat flux"

HEAT = {
    "flux": """  Heat Flux BC = Logical True
  Heat Flux = Variable Velocity 1, Velocity 2, Velocity 3, Temperature
    Real Lua "frictionflux(tx[0], tx[1], tx[2], tx[3])\"""",
    "loads": """  Heat Flux BC = Logical True
  Heat Flux = Real #G
  Temperature Load = Variable Flow Solution Loads 1, Flow Solution Loads 2, Flow Solution Loads 3, Velocity 1, Velocity 2, Velocity 3, Coordinate 2
    Real Lua "frictionloads(tx[0], tx[1], tx[2], tx[3], tx[4], tx[5], tx[6])\"""",
    "geo": """  Heat Flux BC = Logical True
  Heat Flux = Real #G""",
}


def case(name, parts="serial", bc="periodic", heat="flux", halve=False, ls_heat="direct", ls_stokes="direct",
         element="P2/P1", steps=0, dt=100.0, sliding_dT=0.0, out_every=10, Ts=243.15, Tb0=None,
         strain=False, sh_periodic=True, ls_strain="direct"):
    """steps = 0: steady Stokes + steady heat; steps > 0: transient, one Stokes and one heat solve per step
    (sliding_dT > 0: temperature-dependent sliding, so the two are coupled and errors can grow in time)"""
    return dict(name=name, parts=parts, bc=bc, heat=heat, halve=halve, ls_heat=ls_heat, ls_stokes=ls_stokes,
                steps=steps, dt=dt, sliding_dT=sliding_dT, out_every=out_every, Ts=Ts,
                Tb0=Ts + 20.0 if Tb0 is None else Tb0, strain=strain, sh_periodic=sh_periodic, ls_strain=ls_strain,
                element=element)


def physics_key(c):
    """cases with the same physics share a reference (serial, direct solvers)"""
    return (c["bc"], c["heat"], c["halve"], c["element"], c["steps"], c["dt"], c["sliding_dT"], c["Ts"], c["Tb0"],
            c["strain"], c["sh_periodic"] if c["strain"] else None)


CASES = [
    # R: serial references (direct solvers). Their y-uniformity tests the BCs and heating on their own.
    case("R_per_flux"), case("R_per_loads", heat="loads"), case("R_per_loads_halve", heat="loads", halve=True),
    case("R_per_geo", heat="geo"),
    case("R_wall_flux", bc="wall"), case("R_wall_loads", bc="wall", heat="loads"),
    case("R_wall_loads_halve", bc="wall", heat="loads", halve=True),
    case("R_per_flux_stab", element="Stabilized"), case("R_per_loads_stab", heat="loads", element="Stabilized"),
    # A: periodic BCs and the partitioning, direct solvers, flux heating (no loads involved)
    *[case(f"A_per_flux_{p}", parts=p) for p in ("xcut", "xcut_per", "ycut", "ycut_per", "metis", "metis_per")],
    # T: transient, temperature-dependent sliding (Stokes and heat coupled each step), 100 x 100 yr from a
    #    linear profile 270 K (bed) - 255.15 K (surface), the bed warming towards melting and sliding.
    #    Snapshots every 5 steps (snap_t<step>.bin.<pe>): plot_bed_time.py shows them with a time slider.
    #    P1/P1 stabilised Stokes: P2/P1 took over an hour per serial run.
    *[case(f"T_{h}_{p}" + ("" if lh == "direct" else "_iter"), parts=p, heat=h, halve=(h == "loads"),
           steps=100, dt=100.0, out_every=5, sliding_dT=1.0, Ts=255.15, Tb0=270.0, element="Stabilized",
           ls_heat=lh, ls_stokes=ls)
      for h in ("flux", "loads")
      for p, lh, ls in (("serial", "direct", "direct"), ("ycut_per", "direct", "direct"),
                        ("ycut_per", "gcr_ilu2", "gcr_ilu8"), ("metis_per", "gcr_ilu2", "gcr_ilu8"))],
    # S: as T, but with the ElmerStreams sliding law (dT = delta Tscale = 0.15 K) and strain heating
    #    (DeformationalHeatSolver + Heat Source), from 271 K at the bed, 257.15 K at the surface (the bed
    #    reaches the sliding band near melting during the run). "_iter": the ElmerStreams solvers
    #    (heat GCR+ILU2, Stokes GCR+ILU8, strain heating GCR+ILU1). "_shnoper": strain heating without
    #    Periodic BC Strain Heating. "_nostrain": the sharp sliding law alone.
    *[case(f"S_{h}_{p}" + ("" if it is None else "_iter") + sfx, parts=p, heat=h, halve=(h == "loads"),
           steps=100, dt=100.0, out_every=5, sliding_dT=0.15, Ts=257.15, Tb0=271.0, element="Stabilized",
           strain=st, sh_periodic=shp,
           ls_heat="direct" if it is None else "gcr_ilu2", ls_stokes="direct" if it is None else "gcr_ilu8",
           ls_strain="direct" if it is None else "gcr_ilu1")
      for h in ("loads", "flux")
      for p, it, st, shp, sfx in (("serial", None, True, True, ""), ("ycut_per", None, True, True, ""),
                                  ("ycut_per", 1, True, True, ""), ("metis_per", 1, True, True, ""),
                                  ("serial", None, True, False, "_shnoper"), ("ycut_per", 1, True, False, "_shnoper"),
                                  ("serial", None, False, True, "_nostrain"), ("ycut_per", 1, False, True, "_nostrain"))],
    # W: as S (loads heating), but warm: 259.15 K surface, 272.5 K initial bed, so the bed warms through
    #    the sharp sliding band (dT = 0.15 K) during the run and the sliding / frictional-heating feedback
    #    switches on (in S the bed stayed ~1.5 K below melting: mean sliding 0.003 m/a)
    *[case(f"W_loads_{p}" + ("" if it is None else "_iter") + sfx, parts=p, heat="loads", halve=True,
           steps=100, dt=100.0, out_every=5, sliding_dT=0.15, Ts=259.15, Tb0=272.5, element="Stabilized",
           strain=st, ls_heat="direct" if it is None else "gcr_ilu2",
           ls_stokes="direct" if it is None else "gcr_ilu8", ls_strain="direct" if it is None else "gcr_ilu1")
      for p, it, st, sfx in (("serial", None, True, ""), ("ycut_per", None, True, ""), ("ycut_per", 1, True, ""),
                             ("metis_per", 1, True, ""), ("serial", None, False, "_nostrain"),
                             ("ycut_per", 1, False, "_nostrain"))],
    # G: the ElmerStreams combination on partitionings made WITHOUT ElmerGrid's -periodic
    case("G_streams_ycut_noper", parts="ycut", heat="loads", halve=True, ls_heat="gcr_ilu2", ls_stokes="gcr_ilu8"),
    case("G_streams_metis_noper", parts="metis", heat="loads", halve=True, ls_heat="gcr_ilu2", ls_stokes="gcr_ilu8"),
    case("A_per_geo_ycut_per", parts="ycut_per", heat="geo"),
    # B: nodal friction loads in parallel, direct solvers
    *[case(f"B_per_loads_{p}", parts=p, heat="loads") for p in ("xcut", "ycut_per", "metis_per")],
    case("B_per_loads_halve_ycut_per", parts="ycut_per", heat="loads", halve=True),
    case("B_wall_loads_ycut", parts="ycut", bc="wall", heat="loads"),
    case("B_wall_loads_metis", parts="metis", bc="wall", heat="loads"),
    # C: the heat solver's iterative method / preconditioner (flux heating, periodic, y-cuts)
    *[case(f"C_per_flux_ycut_per_{ls}", parts="ycut_per", ls_heat=ls) for ls in ("gcr_ilu2", "bicgstabl_ilu1", "gcr_none")],
    case("C_wall_flux_ycut_gcr_ilu2", parts="ycut", bc="wall", ls_heat="gcr_ilu2"),
    case("C_per_flux_xcut_gcr_ilu2", parts="xcut", ls_heat="gcr_ilu2"),
    # D: the Stokes solver of ElmerStreams (GCR + ILU8, not aborting when not converged)
    case("D_per_flux_ycut_per_stokesgcr", parts="ycut_per", ls_stokes="gcr_ilu8"),
    case("D_per_loads_ycut_per_stokesgcr", parts="ycut_per", heat="loads", ls_stokes="gcr_ilu8"),
    # E: Stokes element: P1/P1 stabilised instead of P2/P1 under periodic BCs
    case("E_per_flux_ycut_per_stab", parts="ycut_per", element="Stabilized"),
    case("E_per_loads_ycut_per_stab", parts="ycut_per", heat="loads", element="Stabilized"),
    # F: the ElmerStreams combination: periodic, y-cuts with -periodic, loads with the y < 1 halving,
    #    GCR + ILU2 heat, GCR + ILU8 Stokes
    case("F_streams", parts="ycut_per", heat="loads", halve=True, ls_heat="gcr_ilu2", ls_stokes="gcr_ilu8"),
    case("F_streams_flux", parts="ycut_per", heat="flux", ls_heat="gcr_ilu2", ls_stokes="gcr_ilu8"),
]


def mesh_dir(parts):
    return os.path.join(OUT, "meshes", parts)       # OUT is out/np<N>


def prepare_mesh(parts, elmergrid="ElmerGrid"):
    d = mesh_dir(parts)
    n, opts = PARTS[parts]
    done = os.path.join(d, "footprint", f"partitioning.{n}") if n > 1 else os.path.join(d, "footprint", "mesh.header")
    if os.path.exists(done):
        return d
    shutil.rmtree(d, ignore_errors=True)
    g = GEOM
    write_footprint(os.path.join(d, "footprint"), g["L"], g["W"], g["nx"], g["ny"])
    if n > 1:
        cmd = [elmergrid, "2", "2", "footprint", *opts]
        print("  " + " ".join(cmd), flush=True)
        with open(os.path.join(d, "elmergrid.log"), "w") as f:
            subprocess.run(cmd, cwd=d, check=True, stdout=f, stderr=subprocess.STDOUT)
        if not os.path.isdir(os.path.join(d, "footprint", f"partitioning.{n}")):
            raise SystemExit(f"ElmerGrid did not write {d}/footprint/partitioning.{n} (see {d}/elmergrid.log)")
    return d


def write_case(c):
    d = os.path.join(OUT, c["name"])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    with open(os.path.join(d, "params.lua"), "w") as f:
        f.write(PARAMS.format(W=GEOM["W"], H=GEOM["H"], halve="true" if c["halve"] else "false",
                              sdT=c["sliding_dT"], Ts=repr(c["Ts"]), Tb0=repr(c["Tb0"])))
    with open(os.path.join(HERE, "case.sif.in")) as f:
        sif = f.read()
    per = c["bc"] == "periodic"
    subs = {
        "__MESH_DIR__": mesh_dir(c["parts"]),
        "__NZ1__": str(GEOM["nz_layers"] + 1),
        "__H__": repr(GEOM["H"]),
        "__ESS_LIB__": os.path.join(REPO, "elmer", "SteadyStateIO"),
        "__STOKES_ELEMENT__": c["element"],
        "__LS_STOKES__": LS[c["ls_stokes"]],
        "__LS_HEAT__": LS[c["ls_heat"]],
        "__SIDE1__": "  ! master side of the periodic pair" if per else WALL,
        "__SIDE2__": PERIODIC if per else WALL,
        "__BED_HEAT__": HEAT[c["heat"]],
        "__SIM_TYPE__": ('  Simulation Type = "Steady State"' if not c["steps"] else
                         f'  Simulation Type = "Transient"\n  Timestepping Method = "bdf"\n  BDF Order = 1\n'
                         f'  Timestep Intervals = {c["steps"]}\n  Timestep Sizes = {c["dt"]!r}'),
        "__OUT_EVERY__": str(c["out_every"] if c["steps"] else 1),
        "__NACTIVE__": "5" if c["steps"] else "4",
        "__ACTIVE__": "1 2 3 4 5" if c["steps"] else "1 2 3 4",
        "__STRAIN_EXEC__": '"Always"' if c["strain"] else '"Never"',
        "__LS_STRAIN__": LS[c["ls_strain"]],
        "__HEAT_SOURCE__": ('  ! strain heating (ElmerStreams body force)\n  Heat Source = Variable Strain Heating\n'
                            '    Real Lua "tx[0] / rhoi"' if c["strain"] else "  ! no strain heating"),
        # transient: snapshots of the state every out_every steps, snap_t<step>.bin.<pe> (plot_bed_time.py)
        "__SNAP_SOLVER__": "" if not c["steps"] else f"""Solver 5
  Exec Solver = "After Timestep"
  Exec Intervals = Integer {c["out_every"]}
  Equation = "Snapshot"
  Procedure = "{os.path.join(REPO, "elmer", "SteadyStateIO")}" "StateIO"
  State Mode = String "write"
  State File = File "snap"
  Timestep Suffix = Logical True
  Write Coordinates = Logical True
  State Variable 1 = String "Temperature"
  State Variable 2 = String "Velocity 1"
  State Variable 3 = String "Strain Heating"
End
""",
    }
    subs["__SIDE2__"] = subs["__SIDE2__"].replace("__W__", repr(float(GEOM["W"])))
    subs["__SIDE2__"] = subs["__SIDE2__"].replace(
        "__SH_PERIODIC__", "  Periodic BC Strain Heating = Logical True" if c["strain"] and c["sh_periodic"]
        else "  ! strain heating not periodic" if c["strain"] else "")
    for k, v in subs.items():
        sif = sif.replace(k, v)
    with open(os.path.join(d, "case.sif"), "w") as f:
        f.write(sif)
    with open(os.path.join(d, "ELMERSOLVER_STARTINFO"), "w") as f:
        f.write("case.sif\n1\n")
    return d


def run_case(c, mpirun, timeout):
    d = write_case(c)
    n = PARTS[c["parts"]][0]
    cmd = (mpirun.format(n=n).split() if n > 1 else []) + ["ElmerSolver", "case.sif"]
    t0 = time.time()
    with open(os.path.join(d, "elmer.log"), "w") as f:
        try:
            ret = subprocess.run(cmd, cwd=d, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            ret = "timeout"
    print(f"  {c['name']:38s} {n:2d} procs  exit {ret}  {time.time() - t0:6.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=int(os.environ.get("SLURM_NTASKS", "8")))
    ap.add_argument("--only", nargs="+", help="run cases whose name contains one of these")
    ap.add_argument("--exact", nargs="+", help="run exactly these cases (full names)")
    ap.add_argument("--no-refs", action="store_true",
                    help="do not add the serial references (they ran already, e.g. in another job)")
    ap.add_argument("--no-analyse", action="store_true", help="skip the analysis at the end")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--analyse", action="store_true", help="only analyse finished cases")
    ap.add_argument("--mpirun", default="mpirun -np {n}")
    ap.add_argument("--elmergrid", default="ElmerGrid")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    global PARTS, OUT
    PARTS = parts_table(a.np)
    OUT = os.path.join(OUT, f"np{a.np}")                 # runs with different N do not overwrite each other
    if a.np > GEOM["nx"]:
        print(f"note: xcut cases need --np <= {GEOM['nx']} (footprint columns); they are skipped")
    cases = [c for c in CASES if (not a.only or any(s in c["name"] for s in a.only))
             and (not a.exact or c["name"] in a.exact)
             and not (c["parts"].startswith("xcut") and a.np > GEOM["nx"])]
    if a.list:
        for c in CASES:
            print(f"{c['name']:38s} parts {c['parts']:10s} bc {c['bc']:8s} heat {c['heat']:5s} halve {c['halve']!s:5s} "
                  f"heat-ls {c['ls_heat']:14s} stokes-ls {c['ls_stokes']:8s} {c['element']}")
        return
    if not a.analyse:
        lib = os.path.join(REPO, "elmer", "SteadyStateIO.so")
        if not os.path.exists(lib):
            raise SystemExit(f"{lib} missing: build it with `make -C elmer` (module load elmerfem first)")
        refs = set() if a.no_refs else {physics_key(c) for c in cases}   # every case gets its reference
        is_ref = lambda c: c["parts"] == "serial" and c["ls_heat"] == "direct" and c["ls_stokes"] == "direct"
        cases = [c for c in CASES if is_ref(c) and physics_key(c) in refs
                 and c not in cases] + cases
        for c in cases:
            if PARTS[c["parts"]][0] > a.np:
                raise SystemExit(f"{c['name']} needs {PARTS[c['parts']][0]} tasks, have --np {a.np}")
        print("partitioning", flush=True)
        for p in sorted({c["parts"] for c in cases}):
            prepare_mesh(p, a.elmergrid)
        print("running", flush=True)
        for c in cases:
            run_case(c, a.mpirun, a.timeout)
    if a.no_analyse:
        return
    from analyse import analyse_all
    analyse_all(CASES, PARTS, physics_key, OUT, GEOM)


if __name__ == "__main__":
    main()
