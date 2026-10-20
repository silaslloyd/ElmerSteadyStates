#!/usr/bin/env python3
"""
Periodic-BC / partitioning test matrix: a parallel-sided slab on a flat bed, square footprint, periodic on
all four sides, gravity tilted along x (README.md).

Run inside a batch job with enough MPI tasks (periodic_test/run.cmd):
    python3 periodic_test/run_tests.py --np 16                  # every case
    python3 periodic_test/run_tests.py --np 16 --only R_ A_     # cases whose name contains R_ or A_
    python3 periodic_test/run_tests.py --list                   # print the cases
    python3 periodic_test/run_tests.py --np 16 --analyse        # analysis only (finished cases)

Each case runs in periodic_test/out_sq/np<N>[tags]/<case>/ (case.sif, params.lua, elmer.log, out.bin.<pe>,
snapshots snap_t<step>.<pe>); the table goes to summary.csv there and is printed.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out_sq")
sys.path.insert(0, HERE)
from make_mesh import write_footprint  # noqa: E402

# ---------------------------------------------------------------- geometry and physics (MPa - a - m)
GEOM = dict(L=200e3, W=200e3, H=600.0, nx=40, ny=40, nz_layers=10)   # 200 km square, 5 km cells by default
PARAMS = """
yearinsec = 31556926.0
rhoi = 917.0e-6 / yearinsec^2              -- MPa a2 m-2
gravity = 9.81 * yearinsec^2               -- m a-2
sina = 0.005                               -- slope: gravity tilted along x, z normal to the bed
cosa = math.sqrt(1 - sina^2)
eta = 3.17                                 -- MPa a (1e14 Pa s), Newtonian
Cslip = 1.5e-3                             -- MPa a m-1: tau_b = C u_b at melting
heatconductivity = 2.1 * yearinsec * 1.0e-6
heatcapacity = 2009.0 * yearinsec^2
G = 0.05 * yearinsec * 1.0e-6              -- geothermal flux (0.05 W m-2)
Ts = {Ts}                                  -- surface temperature (K)
Tb0 = {Tb0}                                -- initial basal temperature (K), linear profile to Ts
L = {L}
W = {W}
H = {H}
halve = {halve}                            -- halve the nodal friction load on the master sides x < 1, y < 1
sliding_dT = {sdT}                         -- 0: linear sliding; > 0: Cslip exp(-min(T - T0, 0) / sliding_dT)

function max(a, b) if a > b then return a else return b end end

-- basal drag coefficient: constant, or temperature dependent as ElmerStreams (capped at melting)
function slip(T)
  if sliding_dT <= 0 then return Cslip end
  local d = T - 273.15
  if d > 0 then d = 0 end
  return Cslip * math.exp(-d / sliding_dT)
end

-- frictional heating as nodal loads (ElmerStreams utils.lua frictionloads). With periodic sides the
-- master nodes (x = 0, y = 0) carry the loads of their images too, so they are halved there
-- (both halvings at the corner, x = y = 0)
function frictionloads(lx, ly, lz, vx, vy, vz, x, y)
  local load = max(-lx * vx, 0) + max(-ly * vy, 0) + max(-lz * vz, 0)
  if halve and x < 1 then load = load / 2 end
  if halve and y < 1 then load = load / 2 end
  return load
end

-- frictional heating as a flux: tau_b . u_b = C |u_b|^2, plus geothermal
function frictionflux(vx, vy, vz, T)
  return G + slip(T) * (vx * vx + vy * vy + vz * vz)
end
"""

WARM = dict(sliding_dT=0.15, Ts=259.15, Tb0=272.5)   # the bed warms through the sharp sliding band


# ---------------------------------------------------------------- partitionings of the footprint
def box_layout(N, L, W):
    """px x py = N boxes, as close to square as the factors of N allow"""
    best = None
    for px in range(1, N + 1):
        if N % px:
            continue
        py = N // px
        score = abs(np.log((L / px) / (W / py)))
        if best is None or score < best[0]:
            best = (score, px, py)
    return best[1], best[2]


def parts_table(N, box=None):
    px, py = box or box_layout(N, GEOM["L"], GEOM["W"])
    if px * py != N:
        raise SystemExit(f"--box {px} {py}: needs px * py = --np ({N})")
    per = ["-periodic", "1", "1", "0"]
    return {
        "serial":    (1, []),
        "box_per":   (N, ["-partition", str(px), str(py), "1", *per]),     # boxes, px along x by py along y
        "xcut_per":  (N, ["-partition", str(N), "1", "1", *per]),          # strips: cuts across x only
        "ycut_per":  (N, ["-partition", "1", str(N), "1", *per]),          # strips: cuts across y only
        "metis_per": (N, ["-metiskway", str(N), *per]),
        "box_noper": (N, ["-partition", str(px), str(py), "1"]),           # boxes without -periodic
    }


PARTS = parts_table(16)

# ---------------------------------------------------------------- linear solvers
LS = {
    "direct": """  Linear System Solver = Direct
  Linear System Direct Method = MUMPS
  Mumps Percentage Increase Working Space = Integer 200
  Linear System Convergence Tolerance = 1.0e-12   ! FreeSurfaceSolver requires it even with Direct""",
    # ElmerStreams heat solver
    "gcr_ilu2": """  Linear System Solver = "Iterative"
  Linear System Iterative Method = "GCR"
  Linear System GCR Restart = Integer 200
  Linear System Max Iterations = 15000
  Linear System Convergence Tolerance = 1.0E-08
  Linear System Abort Not Converged = True
  Linear System Preconditioning = "ILU2"
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
    # ElmerStreams free-surface solver
    "bicgstab_ilu0": """  Linear System Solver = Iterative
  Linear System Max Iterations = 1500
  Linear System Iterative Method = BiCGStab
  Linear System Preconditioning = ILU0
  Linear System Convergence Tolerance = Real 1.0e-6
  Linear System Abort Not Converged = False
  Linear System Residual Output = 1""",
}
ITER = dict(ls_heat="gcr_ilu2", ls_stokes="gcr_ilu8", ls_fs="bicgstab_ilu0")      # the ElmerStreams solvers

HEAT = {
    "flux": """  Heat Flux BC = Logical True
  Heat Flux = Variable Velocity 1, Velocity 2, Velocity 3, Temperature
    Real Lua "frictionflux(tx[0], tx[1], tx[2], tx[3])\"""",
    "loads": """  Heat Flux BC = Logical True
  Heat Flux = Real #G
  Temperature Load = Variable Flow Solution Loads 1, Flow Solution Loads 2, Flow Solution Loads 3, Velocity 1, Velocity 2, Velocity 3, Coordinate 1, Coordinate 2
    Real Lua "frictionloads(tx[0], tx[1], tx[2], tx[3], tx[4], tx[5], tx[6], tx[7])\"""",
}

# free surface: set from the command line (--fs-dt, --fs-steps, --fs-every, --nsvec, --fs-lib)
FS = dict(dt=0.5, steps=2000, every=100, nsvec=None, lib="FreeSurfaceSolver")


def case(name, parts="serial", heat="flux", halve=None, steps=0, dt=100.0, out_every=5, sliding_dT=0.0,
         Ts=243.15, Tb0=None, ss_iters=1, fs=False, element="Stabilized", ls_heat="direct", ls_stokes="direct",
         ls_fs="direct", it=False):
    """steps = 0: steady (ss_iters coupled iterations); steps > 0: transient, Stokes then heat each step.
    fs: free surface (transient; dt, steps and snapshot interval from FS). it: the ElmerStreams iterative
    solvers for everything."""
    if it:
        ls_heat, ls_stokes, ls_fs = ITER["ls_heat"], ITER["ls_stokes"], ITER["ls_fs"]
    if fs:
        steps, dt, out_every = FS["steps"], FS["dt"], FS["every"]
    return dict(name=name, parts=parts, heat=heat, halve=(heat == "loads") if halve is None else halve,
                steps=steps, dt=dt, out_every=out_every, sliding_dT=sliding_dT, Ts=Ts,
                Tb0=Ts + 20.0 if Tb0 is None else Tb0, ss_iters=ss_iters, fs=fs, fssa=fs and bool(FS["nsvec"]),
                element=element, ls_heat=ls_heat, ls_stokes=ls_stokes, ls_fs=ls_fs if fs else "-")


def is_direct(c):
    return c["ls_heat"] == "direct" and c["ls_stokes"] == "direct" and c["ls_fs"] in ("direct", "-")


def physics_key(c):
    """cases with the same physics share a reference (serial direct if there is one, else partitioned direct)"""
    return (c["heat"], c["halve"], c["element"], c["steps"], c["dt"], c["sliding_dT"], c["Ts"], c["Tb0"],
            c["ss_iters"], c["fs"], c["fssa"])


def build_cases():
    P = ("box_per", "metis_per", "ycut_per")
    return [
        # R: serial, direct, steady. The y- and x-uniformity of the periodic BCs and the heating alone;
        #    R_loads (no halving) shows whether the master sides need the halving
        case("R_flux"), case("R_loads", heat="loads", halve=False), case("R_loads_halve", heat="loads"),
        # A: the partitionings, direct solvers, steady, flux heating. box_noper: boxes without -periodic
        *[case(f"A_flux_{p}", parts=p) for p in ("box_per", "xcut_per", "ycut_per", "metis_per", "box_noper")],
        # B: nodal friction loads in parallel (halved on the master sides), direct
        *[case(f"B_loads_{p}", parts=p, heat="loads") for p in ("box_per", "metis_per")],
        # C: the ElmerStreams iterative solvers, steady (tolerance-level errors where?)
        *[case(f"C_{h}_{p}_iter", parts=p, heat=h, it=True) for h in ("flux", "loads") for p in P],
        # T: transient, warm, the ElmerStreams sliding law (dT = 0.15 K): 100 x 100 yr from 272.5 K at the
        #    bed, 259.15 K at the surface, the bed warms through the sliding band and the sliding / frictional
        #    heating feedback switches on. Fixed surface. Snapshots every 5 steps.
        *[case(f"T_{h}_serial", heat=h, steps=100, **WARM) for h in ("loads", "flux")],
        *[case(f"T_{h}_box_per", parts="box_per", heat=h, steps=100, **WARM) for h in ("loads", "flux")],
        *[case(f"T_{h}_{p}_iter", parts=p, heat=h, steps=100, it=True, **WARM) for h in ("loads", "flux") for p in P],
        # SS: steady, warm, coupled (up to 60 Picard iterations Stokes -> heat)
        case("SS_loads_serial", heat="loads", ss_iters=60, **WARM),
        *[case(f"SS_loads_{p}_iter", parts=p, heat="loads", ss_iters=60, it=True, **WARM) for p in P],
        # FS: as T with a free surface (Zs periodic too; StructuredMeshMapper moves the mesh before each
        #     step, the ElmerStreams FreeSurfaceSolver with its limiter after it). Zero accumulation, so the
        #     exact surface stays flat at H. Time step from --fs-dt (default 0.5 yr: the free surface is
        #     unstable above about 1 yr without FSSA), --fs-steps steps, snapshots every --fs-every.
        case("FS_loads_serial", heat="loads", fs=True, **WARM),
        case("FS_loads_box_per", parts="box_per", heat="loads", fs=True, **WARM),
        *[case(f"FS_loads_{p}_iter", parts=p, heat="loads", fs=True, it=True, **WARM) for p in P],
        case("FS_flux_box_per_iter", parts="box_per", heat="flux", fs=True, it=True, **WARM),
    ]


CASES = build_cases()


def mesh_dir(parts):
    n, opts = PARTS[parts]
    tag = "_".join(opts[1:3]) if parts.startswith("box") else ""      # the box layout in the name
    return os.path.join(OUT, "meshes", parts + (f"_{tag}" if tag else ""))


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


def solver_blocks(c):
    """the solver sections, numbered in order; returns (text, ice-body solver list, surface solver)"""
    ess = os.path.join(REPO, "elmer", "SteadyStateIO")
    blocks, ice, surf = [], [], None
    fsv = '  State Variable {k} = String "Zs"\n' if c["fs"] else ""

    def add(text, body="ice"):
        nonlocal surf
        k = len(blocks) + 1
        blocks.append(f"Solver {k}\n{text}End\n")
        if body == "ice":
            ice.append(k)
        else:
            surf = k
    if c["fs"]:
        add("""  Equation = "MapCoordinate"
  Procedure = "StructuredMeshMapper" "StructuredMeshMapper"
  Exec Solver = "Before Timestep"
  Active Coordinate = Integer 3
  Minimum Height = Real 1.0
  Variable = -nooutput DummyR
  Mesh Velocity Variable = String "dSdt"
  Mesh Update Variable = String "dS"
  Mesh Velocity First Zero = Logical True
  Exported Variable 1 = -dofs 1 "dS"
  Exported Variable 2 = -dofs 1 "dSdt"
  Adjust Surface = Logical True
  Bubbles in global system = False
""")
    add(f"""  Equation = "Stokes-Vec"
  Procedure = "{FS['nsvec'] or 'IncompressibleNSVec'}" "IncompressibleNSSolver"
  Div-Curl Discretization = Logical False
  Stokes Flow = Logical True
  Flow Model = Stokes
  Stabilization Method = String "{c['element']}"
  Optimize Bandwidth = Logical True
  ! Newtonian viscosity and a linear sliding law: one iteration solves it
  Nonlinear System Max Iterations = 1
  Nonlinear System Min Iterations = 1
  Nonlinear System Convergence Tolerance = 1.0e-8
  Calculate Loads = Logical True
  Exported Variable 1 = -dofs 4 Flow Solution Loads
  Steady State Convergence Tolerance = 1.0e-9
{LS[c['ls_stokes']]}
""")
    add(f"""  Equation = String "Heat Equation"
  Procedure = File "HeatSolveVec" "HeatSolver"
  Variable = String "Temperature"
  Stabilize = Logical True
  Optimize Bandwidth = Logical True
  Nonlinear System Max Iterations = 1
  Nonlinear System Min Iterations = 1
  Nonlinear System Convergence Tolerance = 1.0e-8
  Steady State Convergence Tolerance = 1.0e-9
{LS[c['ls_heat']]}
""")
    add(f"""  Exec Solver = "After All"
  Equation = "StateWrite"
  Procedure = "{ess}" "StateIO"
  State Mode = String "write"
  State File = File "out.bin"
  Write Coordinates = Logical True
  State Variable 1 = String "Temperature"
  State Variable 2 = String "Velocity 1"
  State Variable 3 = String "Velocity 2"
  State Variable 4 = String "Velocity 3"
  State Variable 5 = String "Pressure"
  State Variable 6 = String "Flow Solution Loads 1"
  State Variable 7 = String "Flow Solution Loads 2"
  State Variable 8 = String "Flow Solution Loads 3"
{fsv.format(k=9)}""")
    if c["steps"]:
        # snapshots every out_every steps, snap_t<step>.<pe> (plot_bed_time.py); before the free-surface
        # update, so Zs and the coordinates belong together
        add(f"""  Exec Solver = "After Timestep"
  Exec Intervals = Integer {c['out_every']}
  Equation = "Snapshot"
  Procedure = "{ess}" "StateIO"
  State Mode = String "write"
  State File = File "snap"
  Timestep Suffix = Logical True
  Write Coordinates = Logical True
  State Variable 1 = String "Temperature"
  State Variable 2 = String "Velocity 1"
  State Variable 3 = String "Velocity 2"
{fsv.format(k=4)}""")
    if c["fs"]:
        # the ElmerStreams free-surface solver (Solver 13 of ThermoCoupled_template.sif)
        add(f"""  Equation = "Free Surface Top"
  Exec Solver = "After Timestep"
  Procedure = "{FS['lib']}" "FreeSurfaceSolver"
  Variable = "Zs"
  Variable DOFs = 1
  Exported Variable 1 = -dofs 1 "Zs Residual"
{LS[c['ls_fs']]}
  Nonlinear System Max Iterations = 100
  Nonlinear System Convergence Tolerance = 1.0e-5
  Nonlinear System Relaxation Factor = 1.00
  Steady State Convergence Tolerance = 1.0e-04
  Stabilization Method = Bubbles
  Apply Dirichlet = Logical True
""", body="surface")
    return "\n".join(blocks), ice, surf


def write_case(c):
    d = os.path.join(OUT, c["name"])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    g = GEOM
    with open(os.path.join(d, "params.lua"), "w") as f:
        f.write(PARAMS.format(L=g["L"], W=g["W"], H=g["H"], halve="true" if c["halve"] else "false",
                              sdT=c["sliding_dT"], Ts=repr(c["Ts"]), Tb0=repr(c["Tb0"])))
    with open(os.path.join(HERE, "case.sif.in")) as f:
        sif = f.read()
    solvers, ice, surf = solver_blocks(c)
    pvars = ["Velocity 1", "Velocity 2", "Velocity 3", "Pressure", "Temperature"] + (["Zs"] if c["fs"] else [])
    fssa = """  FSSA Theta = Real 1.0
  FSSA Flag = String "full"
  FSSA Accumulation = Real 0.0
""" if c["fssa"] else ""
    subs = {
        "__MESH_DIR__": mesh_dir(c["parts"]),
        "__NZ1__": str(g["nz_layers"] + 1),
        "__H__": repr(g["H"]),
        "__L__": repr(float(g["L"])),
        "__W__": repr(float(g["W"])),
        "__SIM_TYPE__": ('  Simulation Type = "Steady State"' if not c["steps"] else
                         f'  Simulation Type = "Transient"\n  Timestepping Method = "bdf"\n  BDF Order = 1\n'
                         f'  Timestep Intervals = {c["steps"]}\n  Timestep Sizes = {c["dt"]!r}'),
        "__OUT_EVERY__": str(c["out_every"] if c["steps"] else 1),
        "__SS_ITERS__": str(c["ss_iters"]),
        "__NACTIVE__": str(len(ice)),
        "__ACTIVE__": " ".join(map(str, ice)),
        "__SOLVERS__": solvers,
        "__PERIODIC_VARS__": "\n".join(f"  Periodic BC {v} = Logical True" for v in pvars),
        "__BED_HEAT__": HEAT[c["heat"]],
        "__FS_MATERIAL__": "  Min Zs = Real 10.0\n  Max Zs = Real 10000.0" if c["fs"] else "",
        "__FS_BED__": "  Bottom Surface = Real 0.0" if c["fs"] else "",
        "__FS_TOP__": (f"  Body Id = 2\n  Top Surface = Equals Zs\n{fssa}" if c["fs"] else ""),
        "__FS_BODIES__": "" if not c["fs"] else f"""
Body 2
  Name = "free surface"
  Equation = 2
  Material = 1
  Body Force = 2
  Initial Condition = 2
End

Equation 2
  Name = "free surface"
  Active Solvers(1) = {surf}
  Flow Solution Name = String "Flow Solution"
  Convection = String "Computed"
End

Initial Condition 2
  Zs = Equals Coordinate 3
End

Body Force 2
  Zs Accumulation = Real 0.0
End
""",
    }
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
    print(f"  {c['name']:30s} {n:3d} procs  exit {ret}  {time.time() - t0:8.1f} s", flush=True)


def main():
    global PARTS, OUT, CASES
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=int(os.environ.get("SLURM_NTASKS", "16")))
    ap.add_argument("--only", nargs="+", help="run cases whose name contains one of these")
    ap.add_argument("--exact", nargs="+", help="run exactly these cases (full names)")
    ap.add_argument("--no-refs", action="store_true",
                    help="do not add the serial references (they ran already, or are too slow: the analysis "
                         "then uses a partitioned direct-solver run of the same physics)")
    ap.add_argument("--no-analyse", action="store_true", help="skip the analysis at the end")
    ap.add_argument("--res", type=float, help="footprint cell size (m), e.g. 1000; default 5000")
    ap.add_argument("--size", type=float, help="side of the square footprint (m), default 200000")
    ap.add_argument("--layers", type=int, help="vertical layers, default 10")
    ap.add_argument("--box", nargs=2, type=int, metavar=("PX", "PY"),
                    help="box partitions: PX along x by PY along y (PX * PY = --np); default about square")
    ap.add_argument("--fs-dt", type=float, help=f"free-surface cases: time step (yr), default {FS['dt']}")
    ap.add_argument("--fs-steps", type=int, help=f"free-surface cases: steps, default {FS['steps']}")
    ap.add_argument("--fs-every", type=int, help=f"free-surface cases: snapshot interval, default {FS['every']}")
    ap.add_argument("--nsvec", help="Stokes library with FSSA (e.g. ElmerStreams SRC/IncompressibleNSVec.so): "
                                    "used for every case, and the free surface gets FSSA (theta 1, full)")
    ap.add_argument("--fs-lib", help="free-surface library (e.g. ElmerStreams SRC/FreeSurfaceSolver.so); "
                                     "default the Elmer module FreeSurfaceSolver")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--analyse", action="store_true", help="only analyse finished cases")
    ap.add_argument("--mpirun", default="mpirun -np {n}")
    ap.add_argument("--elmergrid", default="ElmerGrid")
    ap.add_argument("--timeout", type=float, default=None, help="per case (s), default none")
    a = ap.parse_args()
    tag = ""
    if a.size:
        GEOM["L"] = GEOM["W"] = a.size
        tag += f"_L{a.size / 1e3:g}km"
    res = a.res or 5000.0
    GEOM["nx"] = max(1, int(round(GEOM["L"] / res)))
    GEOM["ny"] = max(1, int(round(GEOM["W"] / res)))
    if a.res:
        tag += f"_dx{a.res:g}"
    if a.layers:
        GEOM["nz_layers"] = a.layers
        tag += f"_z{a.layers}"
    if a.fs_dt:
        FS["dt"] = a.fs_dt
        tag += f"_fsdt{a.fs_dt:g}"
    if a.fs_steps:
        FS["steps"] = a.fs_steps
    if a.fs_every:
        FS["every"] = a.fs_every
    if a.nsvec:
        FS["nsvec"] = os.path.abspath(a.nsvec)
        tag += "_fssa"
    if a.fs_lib:
        FS["lib"] = os.path.abspath(a.fs_lib)
    CASES = build_cases()
    OUT = os.path.join(OUT, f"np{a.np}{tag}")
    PARTS = parts_table(a.np, a.box)
    print(f"box partitions: {PARTS['box_per'][1][1]} along x by {PARTS['box_per'][1][2]} along y", flush=True)
    print(f"mesh: {GEOM['nx']} x {GEOM['ny']} cells ({GEOM['L'] / GEOM['nx']:.0f} m x {GEOM['W'] / GEOM['ny']:.0f} m), "
          f"{GEOM['nz_layers']} layers; free surface: dt {FS['dt']:g} yr x {FS['steps']} steps"
          f"{', FSSA' if FS['nsvec'] else ''}; results in {OUT}", flush=True)
    cases = [c for c in CASES if (not a.only or any(s in c["name"] for s in a.only))
             and (not a.exact or c["name"] in a.exact)
             and not (c["parts"] == "xcut_per" and a.np > GEOM["nx"])
             and not (c["parts"] == "ycut_per" and a.np > GEOM["ny"])]
    if a.list:
        for c in CASES:
            print(f"{c['name']:24s} parts {c['parts']:10s} heat {c['heat']:5s} halve {c['halve']!s:5s} "
                  f"steps {c['steps']:5d} x {c['dt']:g} yr  fs {c['fs']!s:5s} heat-ls {c['ls_heat']:8s} "
                  f"stokes-ls {c['ls_stokes']:8s} fs-ls {c['ls_fs']}")
        return
    if not a.analyse:
        lib = os.path.join(REPO, "elmer", "SteadyStateIO.so")
        if not os.path.exists(lib):
            raise SystemExit(f"{lib} missing: build it with `make -C elmer` (module load elmerfem first)")
        refs = set() if a.no_refs else {physics_key(c) for c in cases}   # every case gets its serial reference
        cases = [c for c in CASES if c["parts"] == "serial" and is_direct(c) and physics_key(c) in refs
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
    analyse_all(CASES, physics_key, is_direct, OUT, GEOM)


if __name__ == "__main__":
    main()
