#!/usr/bin/env python3
"""
Periodic-sidewall loads test (README_loads.md): are Elmer's nodal Stokes loads ("Flow Solution Loads", used by
the grounding-line contact and the nodal friction heating) right on periodic boundaries, and if not, by how much?

A parallel-sided slab on a flat bed, flowline footprint L x W, periodic in x (so nothing depends on x), the sides
y = 0 / y = W either periodic (y = W the image of y = 0) or free-slip walls (Velocity 2 = 0, the control), driven
along x by gravity tilted by the slope. Stokes (Newtonian, linear sliding) only, optionally with a free surface.
The exact nodal loads are known: at the bed rho g H (sin a, 0, cos a) (times the sign of Elmer's loads) times the
node's own share of the bed (the area of the adjacent bed faces that touch it, a quarter of each), zero elsewhere.

Run inside a batch job with enough MPI tasks (periodic_test/run_loads.cmd):
    python3 periodic_test/run_loads.py --np 4               # every case, then the table
    python3 periodic_test/run_loads.py --list
    python3 periodic_test/run_loads.py --np 4 --analyse     # the table again
Cases run in periodic_test/out_loads/<case>/; the table goes to out_loads/summary.csv and is printed.
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out_loads")
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
from make_mesh import write_footprint  # noqa: E402
from ess.statefile import read_state_file  # noqa: E402

# ---------------------------------------------------------------- geometry and physics (MPa - a - m)
GEOM = dict(L=100e3, dx=2000.0, dy=2000.0, H=600.0, nz_layers=10)
PARAMS = """
yearinsec = 31556926.0
rhoi = 917.0e-6 / yearinsec^2              -- MPa a2 m-2
gravity = 9.81 * yearinsec^2               -- m a-2
sina = 0.005                               -- slope: gravity tilted along x, z normal to the bed
cosa = math.sqrt(1 - sina^2)
eta = 3.17                                 -- MPa a (1e14 Pa s), Newtonian
Cslip = 1.5e-3                             -- MPa a m-1: tau_b = C u_b
"""
RHOG = 917.0e-6 / 31556926.0 ** 2 * 9.81 * 31556926.0 ** 2      # rho_i g in MPa / m
SINA = 0.005
COSA = float(np.sqrt(1 - SINA ** 2))

FS = dict(dt=0.5, steps=20)          # free-surface cases: 20 x 0.5 yr (the slab's surface is unstable above ~1 yr)

DIRECT = """  Linear System Solver = Direct
  Linear System Direct Method = MUMPS
  Linear System Convergence Tolerance = 1.0e-12"""


def case(name, ny=1, sides="periodic", parts="serial", fs=False, element="Stabilized"):
    return dict(name=name, ny=ny, sides=sides, parts=parts, fs=fs, element=element)


def build_cases():
    cs = []
    for ny in (1, 2, 4):
        for sides in ("periodic", "walls"):
            for parts in ("serial", "xcut"):
                cs.append(case(f"ny{ny}_{sides}_{parts}", ny=ny, sides=sides, parts=parts))
    cs.append(case("ny4_periodic_metis", ny=4, parts="metis"))
    for ny in (1, 2):
        for parts in ("serial", "xcut"):
            cs.append(case(f"fs_ny{ny}_periodic_{parts}", ny=ny, parts=parts, fs=True))
    cs.append(case("fs_ny1_walls_serial", ny=1, sides="walls", fs=True))
    return cs


CASES = build_cases()
NP = 4


def parts_opts(parts, sides):
    per = ["-periodic", "1", "1" if sides == "periodic" else "0", "0"]
    return {"serial": (1, []),
            "xcut": (NP, ["-partition", str(NP), "1", "1", *per]),
            "metis": (NP, ["-metiskway", str(NP), *per])}[parts]


def mesh_dir(c):
    return os.path.join(OUT, "meshes", f"ny{c['ny']}_{c['parts']}_{'per' if c['sides'] == 'periodic' else 'wall'}")


def prepare_mesh(c, elmergrid):
    d = mesh_dir(c)
    n, opts = parts_opts(c["parts"], c["sides"])
    done = os.path.join(d, "footprint", f"partitioning.{n}") if n > 1 else os.path.join(d, "footprint", "mesh.header")
    if os.path.exists(done):
        return d
    shutil.rmtree(d, ignore_errors=True)
    g = GEOM
    nx = int(round(g["L"] / g["dx"]))
    write_footprint(os.path.join(d, "footprint"), g["L"], c["ny"] * g["dy"], nx, c["ny"])
    if n > 1:
        cmd = [elmergrid, "2", "2", "footprint", *opts]
        with open(os.path.join(d, "elmergrid.log"), "w") as f:
            subprocess.run(cmd, cwd=d, check=True, stdout=f, stderr=subprocess.STDOUT)
        if not os.path.isdir(os.path.join(d, "footprint", f"partitioning.{n}")):
            raise SystemExit(f"ElmerGrid did not write {d}/footprint/partitioning.{n} (see {d}/elmergrid.log)")
    return d


def solvers(c):
    ess = os.path.join(REPO, "elmer", "SteadyStateIO")
    blocks, ice, surf = [], [], None

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
""")
    add(f"""  Equation = "Stokes-Vec"
  Procedure = "IncompressibleNSVec" "IncompressibleNSSolver"
  Div-Curl Discretization = Logical False
  Stokes Flow = Logical True
  Flow Model = Stokes
  Stabilization Method = String "{c['element']}"
  Optimize Bandwidth = Logical True
  Nonlinear System Max Iterations = 1
  Nonlinear System Min Iterations = 1
  Nonlinear System Convergence Tolerance = 1.0e-8
  Steady State Convergence Tolerance = 1.0e-9
{LOADS_KW}
{DIRECT}
""")
    LOADS_KW = loads_keywords()
    add(f"""  Exec Solver = "After All"
  Equation = "StateWrite"
  Procedure = "{ess}" "StateIO"
  State Mode = String "write"
  State File = File "out.bin"
  Write Coordinates = Logical True
  State Variable 1 = String "Velocity 1"
  State Variable 2 = String "Velocity 2"
  State Variable 3 = String "Velocity 3"
  State Variable 4 = String "Pressure"
{state_vars(c)}""")
    if c["fs"]:
        add(f"""  Equation = "Free Surface Top"
  Exec Solver = "After Timestep"
  Procedure = "FreeSurfaceSolver" "FreeSurfaceSolver"
  Variable = "Zs"
  Variable DOFs = 1
  Exported Variable 1 = -dofs 1 "Zs Residual"
{DIRECT}
  Nonlinear System Max Iterations = 100
  Nonlinear System Convergence Tolerance = 1.0e-5
  Steady State Convergence Tolerance = 1.0e-04
  Stabilization Method = Bubbles
  Apply Dirichlet = Logical True
""", body="surface")
    return "\n".join(blocks), ice, surf


# what the Stokes solver computes and StateWrite writes; --no-weights / --no-loads take parts out
OPTS = dict(weights=True, loads=True)


def loads_keywords():
    kw = []
    if OPTS["loads"]:
        kw += ["  Calculate Loads = Logical True", "  Exported Variable 1 = -dofs 4 Flow Solution Loads"]
    if OPTS["weights"]:
        kw += ["  Calculate Boundary Weights = Logical True",
               f"  Exported Variable {len(kw) // 2 + 1} = -dofs 1 Flow Solution Boundary Weights"]
    return "\n".join(kw)


def state_vars(c):
    names = ["Velocity 1", "Velocity 2", "Velocity 3", "Pressure"]
    if OPTS["loads"]:
        names += [f"Flow Solution Loads {k}" for k in (1, 2, 3, 4)]
    if OPTS["weights"]:
        names.append("Flow Solution Boundary Weights")
    if c["fs"]:
        names.append("Zs")
    return "".join(f'  State Variable {i} = String "{n}"\n' for i, n in enumerate(names, 1))


def write_case(c):
    d = os.path.join(OUT, c["name"])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    g = GEOM
    with open(os.path.join(d, "params.lua"), "w") as f:
        f.write(PARAMS)
    sif = open(os.path.join(HERE, "loads.sif.in")).read()
    text, ice, surf = solvers(c)
    pvars = ["Velocity 1", "Velocity 2", "Velocity 3", "Pressure"] + (["Zs"] if c["fs"] else [])
    pv = "\n".join(f"  Periodic BC {v} = Logical True" for v in pvars)
    W = c["ny"] * g["dy"]
    if c["sides"] == "periodic":
        y0, yw = "", f"  Periodic BC = 3\n  Periodic BC Translate(3) = Real 0.0 {W!r} 0.0\n{pv}"
    else:
        y0 = yw = "  Velocity 2 = Real 0.0"
    subs = {
        "__MESH_DIR__": mesh_dir(c),
        "__NZ1__": str(g["nz_layers"] + 1),
        "__H__": repr(g["H"]),
        "__L__": repr(float(g["L"])),
        "__SIM_TYPE__": ('  Simulation Type = "Steady State"' if not c["fs"] else
                         f'  Simulation Type = "Transient"\n  Timestepping Method = "bdf"\n  BDF Order = 1\n'
                         f'  Timestep Intervals = {FS["steps"]}\n  Timestep Sizes = {FS["dt"]!r}'),
        "__NACTIVE__": str(len(ice)), "__ACTIVE__": " ".join(map(str, ice)),
        "__SOLVERS__": text,
        "__PERIODIC_VARS__": pv,
        "__Y0__": y0, "__YW__": yw,
        "__FS_MATERIAL__": "  Min Zs = Real 10.0\n  Max Zs = Real 10000.0" if c["fs"] else "",
        "__FS_BED__": "  Bottom Surface = Real 0.0" if c["fs"] else "",
        "__FS_TOP__": "  Body Id = 2\n  Top Surface = Equals Zs" if c["fs"] else "",
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
    open(os.path.join(d, "case.sif"), "w").write(sif)
    open(os.path.join(d, "ELMERSOLVER_STARTINFO"), "w").write("case.sif\n1\n")
    return d


def run_case(c, mpirun, timeout):
    d = write_case(c)
    n = parts_opts(c["parts"], c["sides"])[0]
    cmd = (mpirun.format(n=n).split() if n > 1 else []) + ["ElmerSolver", "case.sif"]
    t0 = time.time()
    with open(os.path.join(d, "elmer.log"), "w") as f:
        try:
            ret = subprocess.run(cmd, cwd=d, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            ret = "timeout"
    open(os.path.join(d, "status.txt"), "w").write(f"{ret}\n")
    print(f"  {c['name']:28s} {n:2d} procs  exit {ret}  {time.time() - t0:6.1f} s", flush=True)
    if ret != 0:
        log = open(os.path.join(d, "elmer.log"), errors="replace").read().splitlines()
        hits = [l for l in log if any(k in l.lower() for k in ("error", "fatal", "signal", "warn", "backtrace"))]
        for l in (hits[-8:] + ["    ..."] + log[-12:]):
            print("      | " + l[:160])


# ---------------------------------------------------------------- analysis
def read_copies(d):
    """per-partition records: {name: (gids, values)} for each out.bin.<pe>"""
    files = sorted(glob.glob(os.path.join(d, "out.bin.*")), key=lambda p: int(p.rsplit(".", 1)[1]))
    out = []
    for f in files:
        try:
            out.append({k.lower(): v for k, v in read_state_file(f).items()})
        except (ValueError, IndexError, OSError) as e:
            raise RuntimeError(f"unreadable {os.path.basename(f)} ({e})")
    return out


def own_area(x, y, L, W, dx, dy):
    """the node's own share of the bed: a quarter of each adjacent bed face (no periodic images)"""
    nx_adj = np.where((np.abs(x) < 1) | (np.abs(x - L) < 1), 1, 2)
    ny_adj = np.where((np.abs(y) < 1) | (np.abs(y - W) < 1), 1, 2)
    return nx_adj * ny_adj * dx * dy / 4.0


def analyse_case(c):
    d = os.path.join(OUT, c["name"])
    st = os.path.join(d, "status.txt")
    status = "?"
    if os.path.exists(st):
        with open(st) as f:
            status = f.read().strip()
    if not os.path.isdir(d):
        return None
    if status not in ("0", "?"):            # "?": run before status.txt existed; read it if it is readable
        print(f"  {c['name']}: skipped (Elmer exit {status})")
        return None
    try:
        parts = read_copies(d)
    except RuntimeError as e:
        print(f"  {c['name']}: skipped, {e}")
        return None
    if not parts:
        print(f"  {c['name']}: skipped, no out.bin.*")
        return None
    missing = [n for n in ("flow solution loads 1", "flow solution loads 2", "flow solution loads 3",
                           "flow solution boundary weights") if n not in parts[0]]
    if missing:
        print(f"  {c['name']}: not written: {', '.join(missing)} (present: {', '.join(sorted(parts[0]))})")
    g = GEOM
    L, W, dx, dy = g["L"], c["ny"] * g["dy"], g["dx"], g["dy"]
    # every copy of every node (shared nodes appear once per partition)
    rows = {}
    for pe, rec in enumerate(parts):
        gids = rec["coordinate 1"][0]
        X, Y, Z = (rec[f"coordinate {k}"][1] for k in (1, 2, 3))
        get = lambda name: dict(zip(*rec[name.lower()])) if name.lower() in rec else {}
        Lc = [get(f"Flow Solution Loads {k}") for k in (1, 2, 3)]
        Wt = get("Flow Solution Boundary Weights")
        for i, gid in enumerate(gids):
            r = rows.setdefault(int(gid), dict(x=X[i], y=Y[i], z=Z[i], L=[], w=[]))
            r["L"].append([Lc[k].get(gid, np.nan) for k in range(3)])
            r["w"].append(Wt.get(gid, np.nan))
    gid = np.array(sorted(rows))
    x = np.array([rows[k]["x"] for k in gid]); y = np.array([rows[k]["y"] for k in gid])
    z = np.array([rows[k]["z"] for k in gid])
    ncopy = np.array([len(rows[k]["L"]) for k in gid])
    Lsum = np.array([np.nansum(rows[k]["L"], axis=0) for k in gid])          # summed over partition copies
    Lone = np.array([rows[k]["L"][0] for k in gid])                          # one copy
    wone = np.array([rows[k]["w"][0] for k in gid])
    Lspread = np.array([np.nanmax(np.ptp(np.array(rows[k]["L"]), axis=0)) if len(rows[k]["L"]) > 1 else 0.0
                        for k in gid])
    bed = np.abs(z) < 1e-3
    zs = z.max()
    H = zs                                                    # uniform: the slab thickness (after the free surface)
    top = np.abs(z - zs) < 1e-3
    A = own_area(x, y, L, W, dx, dy)
    # exact bed loads (magnitudes); the sign of Elmer's loads is taken from the data
    ex = dict(x=RHOG * SINA * H * A, z=RHOG * COSA * H * A)
    if np.isnan(Lone[bed]).all():
        return None
    sgn = {}
    for k, j in (("x", 0), ("z", 2)):
        v = Lone[bed & (A == A.max()), j]
        sgn[k] = float(np.sign(np.nanmedian(v))) if np.isfinite(v).any() else 1.0
        sgn[k] = sgn[k] or 1.0
    xper = (np.abs(x) < 1) | (np.abs(x - L) < 1)             # the x-periodic columns (left out of the classes)
    cls = {"y0 (master)": bed & ~xper & (np.abs(y) < 1),
           "yW (image)": bed & ~xper & (np.abs(y - W) < 1),
           "interior": bed & ~xper & (np.abs(y) >= 1) & (np.abs(y - W) >= 1)}
    out = dict(case=c["name"], ny=c["ny"], sides=c["sides"], parts=c["parts"], fs=c["fs"], H=H)
    for name, m in cls.items():
        if not m.any():
            continue
        tag = name.split()[0]
        for comp, j in (("z", 2), ("x", 0)):
            r1 = sgn[comp] * Lone[m, j] / ex[comp][m]
            rs = sgn[comp] * Lsum[m, j] / ex[comp][m]
            out[f"{tag}_{comp}"] = float(np.mean(r1))
            out[f"{tag}_{comp}_sprd"] = float(np.ptp(r1))
            if (ncopy[m] > 1).any():
                out[f"{tag}_{comp}_sum"] = float(np.mean(rs))
        # cross-flow load relative to the node's exact vertical load (exact 0)
        out[f"{tag}_y"] = float(np.mean(Lone[m, 1] / ex["z"][m]))
        out[f"{tag}_w"] = float(np.mean(wone[m] / A[m]))          # Elmer's boundary weight / own area
    # surface nodes: loads should vanish (relative to the bed's vertical load per node)
    ref = float(np.median(ex["z"][bed & (A == A.max())]))
    for name, m in (("top_master", top & (np.abs(y) < 1)), ("top_image", top & (np.abs(y - W) < 1))):
        if m.any():
            out[f"{name}_z"] = float(np.max(np.abs(Lone[m, 2])) / ref)
            out[f"{name}_y"] = float(np.max(np.abs(Lone[m, 1])) / ref)
    out["shared_copy_spread"] = float(np.max(Lspread[bed]) / ref) if bed.any() else 0.0
    return out


def analyse_all(cases):
    res = [r for r in (analyse_case(c) for c in cases) if r]
    if not res:
        print("no results yet")
        return
    keys = list(dict.fromkeys(k for r in res for k in r))
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(res)
    print("\nratios to the exact load (1 = right); _y: cross-flow load / exact vertical load (0 = right);"
          " _w: Elmer boundary weight / own area; _sum: summed over partition copies")
    show = ["y0_z", "yW_z", "interior_z", "y0_x", "yW_x", "y0_y", "yW_y", "y0_w", "yW_w", "y0_z_sum", "yW_z_sum",
            "top_master_z", "top_image_y", "shared_copy_spread"]
    show = [k for k in show if k in keys]
    print(f"{'case':28s} " + " ".join(f"{k:>12s}" for k in show))
    for r in res:
        print(f"{r['case']:28s} " + " ".join(f"{r[k]:12.4g}" if isinstance(r.get(k), float) else f"{'-':>12s}"
                                              for k in show))
    print(f"\n(table also in {os.path.join(OUT, 'summary.csv')})")


def main():
    global NP
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=int(os.environ.get("SLURM_NTASKS", "4")))
    ap.add_argument("--only", nargs="+", help="run cases whose name contains one of these")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--analyse", action="store_true", help="only analyse finished cases")
    ap.add_argument("--element", choices=["Stabilized", "P2/P1"], help="Stokes element (default Stabilized, "
                                                                          "as ElmerStreams)")
    ap.add_argument("--mpirun", default=os.environ.get("ESS_MPIRUN", "mpirun -np {n}"))
    ap.add_argument("--elmergrid", default="ElmerGrid")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--no-weights", action="store_true", help="no Calculate Boundary Weights (crash bisection)")
    ap.add_argument("--no-loads", action="store_true", help="no Calculate Loads (crash bisection)")
    a = ap.parse_args()
    OPTS["weights"] = not a.no_weights
    OPTS["loads"] = not a.no_loads
    NP = min(a.np, 4)
    cases = [c for c in CASES if not a.only or any(s in c["name"] for s in a.only)]
    if a.element:
        for c in cases:
            c["element"] = a.element
    if a.list:
        for c in cases:
            print(f"{c['name']:28s} ny {c['ny']}  sides {c['sides']:8s} parts {c['parts']:6s} fs {c['fs']}")
        return
    if not a.analyse:
        lib = os.path.join(REPO, "elmer", "SteadyStateIO.so")
        if not os.path.exists(lib):
            raise SystemExit(f"{lib} missing: build it with `make -C elmer` (module load elmerfem first)")
        os.makedirs(OUT, exist_ok=True)
        print(f"footprint {GEOM['L'] / 1e3:g} km x ny * {GEOM['dy'] / 1e3:g} km, {GEOM['dx'] / 1e3:g} km cells, "
              f"{GEOM['nz_layers']} layers, H = {GEOM['H']:g} m; results in {OUT}", flush=True)
        for c in cases:
            prepare_mesh(c, a.elmergrid)
            run_case(c, a.mpirun, a.timeout)
    analyse_all(cases)


if __name__ == "__main__":
    main()
