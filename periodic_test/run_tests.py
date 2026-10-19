#!/usr/bin/env python3
"""
Periodic-BC / partitioning test matrix for the thermo-mechanical ice slab (README.md).

Run inside a batch job with enough MPI tasks (jobs: periodic_test/run.cmd):
    python3 periodic_test/run_tests.py --np 8                 # every case
    python3 periodic_test/run_tests.py --np 8 --only A_ B_    # cases whose name contains A_ or B_
    python3 periodic_test/run_tests.py --list                 # print the cases
    python3 periodic_test/run_tests.py --analyse              # analysis only (finished cases)

Each case runs in periodic_test/out/<case>/ (sif, params.lua, elmer.log, out.bin.<pe>);
the summary goes to periodic_test/out/summary.csv and is printed.
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
Ts = 243.15                                -- surface temperature (K)
W = {W}
halve_y0 = {halve}                         -- ElmerStreams frictionloads: halve the load at y < 1 m

function max(a, b) if a > b then return a else return b end end

-- frictional heating as nodal loads (as ElmerStreams utils.lua frictionloads)
function frictionloads(lx, ly, lz, vx, vy, vz, y)
  local load = max(-lx * vx, 0) + max(-ly * vy, 0) + max(-lz * vz, 0)
  if halve_y0 and y < 1 then load = load / 2 end
  return load
end

-- frictional heating as a flux: tau_b . u_b = Cslip |u_b|^2 (+ geothermal)
function frictionflux(vx, vy, vz)
  return G + Cslip * (vx * vx + vy * vy + vz * vz)
end
"""

# ---------------------------------------------------------------- partitionings of the footprint
# name -> (number of partitions, ElmerGrid options)
PARTS = {
    "serial":     (1, []),
    "x8":         (8, ["-partition", "8", "1", "1"]),                          # cuts across x only
    "y4x2":       (8, ["-partition", "2", "4", "1"]),                          # as stream.cmd, no -periodic
    "y4x2_per":   (8, ["-partition", "2", "4", "1", "-periodic", "0", "1", "0"]),   # as stream.cmd
    "metis8":     (8, ["-metiskway", "8"]),
    "metis8_per": (8, ["-metiskway", "8", "-periodic", "0", "1", "0"]),
}

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
  Periodic BC Temperature = Logical True"""
WALL = "  Velocity 2 = Real 0.0        ! free slip; zero heat flux"

HEAT = {
    "flux": """  Heat Flux BC = Logical True
  Heat Flux = Variable Velocity 1, Velocity 2, Velocity 3
    Real Lua "frictionflux(tx[0], tx[1], tx[2])\"""",
    "loads": """  Heat Flux BC = Logical True
  Heat Flux = Real #G
  Temperature Load = Variable Flow Solution Loads 1, Flow Solution Loads 2, Flow Solution Loads 3, Velocity 1, Velocity 2, Velocity 3, Coordinate 2
    Real Lua "frictionloads(tx[0], tx[1], tx[2], tx[3], tx[4], tx[5], tx[6])\"""",
    "geo": """  Heat Flux BC = Logical True
  Heat Flux = Real #G""",
}


def case(name, parts="serial", bc="periodic", heat="flux", halve=False, ls_heat="direct", ls_stokes="direct",
         element="P2/P1"):
    return dict(name=name, parts=parts, bc=bc, heat=heat, halve=halve, ls_heat=ls_heat, ls_stokes=ls_stokes,
                element=element)


def physics_key(c):
    """cases with the same physics share a reference (serial, direct solvers)"""
    return (c["bc"], c["heat"], c["halve"], c["element"])


CASES = [
    # R: serial references (direct solvers). Their y-uniformity tests the BCs and heating on their own.
    case("R_per_flux"), case("R_per_loads", heat="loads"), case("R_per_loads_halve", heat="loads", halve=True),
    case("R_per_geo", heat="geo"),
    case("R_wall_flux", bc="wall"), case("R_wall_loads", bc="wall", heat="loads"),
    case("R_wall_loads_halve", bc="wall", heat="loads", halve=True),
    case("R_per_flux_stab", element="Stabilized"), case("R_per_loads_stab", heat="loads", element="Stabilized"),
    # A: periodic BCs and the partitioning, direct solvers, flux heating (no loads involved)
    *[case(f"A_per_flux_{p}", parts=p) for p in ("x8", "y4x2", "y4x2_per", "metis8", "metis8_per")],
    case("A_per_geo_y4x2_per", parts="y4x2_per", heat="geo"),
    # B: nodal friction loads in parallel, direct solvers
    *[case(f"B_per_loads_{p}", parts=p, heat="loads") for p in ("x8", "y4x2_per", "metis8_per")],
    case("B_per_loads_halve_y4x2_per", parts="y4x2_per", heat="loads", halve=True),
    case("B_wall_loads_y4x2", parts="y4x2", bc="wall", heat="loads"),
    case("B_wall_loads_metis8", parts="metis8", bc="wall", heat="loads"),
    # C: the heat solver's iterative method / preconditioner (flux heating, periodic, y-cuts)
    *[case(f"C_per_flux_y4x2_per_{ls}", parts="y4x2_per", ls_heat=ls) for ls in ("gcr_ilu2", "bicgstabl_ilu1", "gcr_none")],
    case("C_wall_flux_y4x2_gcr_ilu2", parts="y4x2", bc="wall", ls_heat="gcr_ilu2"),
    case("C_per_flux_x8_gcr_ilu2", parts="x8", ls_heat="gcr_ilu2"),
    # D: the Stokes solver of ElmerStreams (GCR + ILU8, not aborting when not converged)
    case("D_per_flux_y4x2_per_stokesgcr", parts="y4x2_per", ls_stokes="gcr_ilu8"),
    case("D_per_loads_y4x2_per_stokesgcr", parts="y4x2_per", heat="loads", ls_stokes="gcr_ilu8"),
    # E: Stokes element: P1/P1 stabilised instead of P2/P1 under periodic BCs
    case("E_per_flux_y4x2_per_stab", parts="y4x2_per", element="Stabilized"),
    case("E_per_loads_y4x2_per_stab", parts="y4x2_per", heat="loads", element="Stabilized"),
    # F: the ElmerStreams combination: periodic, y-cuts with -periodic, loads with the y < 1 halving,
    #    GCR + ILU2 heat, GCR + ILU8 Stokes
    case("F_streams", parts="y4x2_per", heat="loads", halve=True, ls_heat="gcr_ilu2", ls_stokes="gcr_ilu8"),
    case("F_streams_flux", parts="y4x2_per", heat="flux", ls_heat="gcr_ilu2", ls_stokes="gcr_ilu8"),
]


def mesh_dir(parts):
    return os.path.join(OUT, "meshes", parts)


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
        f.write(PARAMS.format(W=GEOM["W"], halve="true" if c["halve"] else "false"))
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
    }
    subs["__SIDE2__"] = subs["__SIDE2__"].replace("__W__", repr(float(GEOM["W"])))
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
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--analyse", action="store_true", help="only analyse finished cases")
    ap.add_argument("--mpirun", default="mpirun -np {n}")
    ap.add_argument("--elmergrid", default="ElmerGrid")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    cases = [c for c in CASES if not a.only or any(s in c["name"] for s in a.only)]
    if a.list:
        for c in CASES:
            print(f"{c['name']:38s} parts {c['parts']:10s} bc {c['bc']:8s} heat {c['heat']:5s} halve {c['halve']!s:5s} "
                  f"heat-ls {c['ls_heat']:14s} stokes-ls {c['ls_stokes']:8s} {c['element']}")
        return
    if not a.analyse:
        lib = os.path.join(REPO, "elmer", "SteadyStateIO.so")
        if not os.path.exists(lib):
            raise SystemExit(f"{lib} missing: build it with `make -C elmer` (module load elmerfem first)")
        refs = {physics_key(c) for c in cases}             # make sure every case has its reference
        cases = [c for c in CASES if c["name"].startswith("R_") and physics_key(c) in refs
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
    from analyse import analyse_all
    analyse_all(CASES, PARTS, physics_key, OUT, GEOM)


if __name__ == "__main__":
    main()
