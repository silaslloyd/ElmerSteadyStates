#!/usr/bin/env python3
"""
Periodic-sidewall loads test (README_loads.md): are Elmer's nodal Stokes loads ("Flow Solution Loads", used by
the grounding-line contact and the nodal friction heating) right on periodic boundaries, and if not, by how much?
And does elmer/FixPeriodicLoads.F90 (a periodic copy of the loads) fix them?

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


def case(name, ny=1, sides="periodic", parts="serial", fs=False, element="Stabilized", shear=False, shift=0):
    return dict(name=name, ny=ny, sides=sides, parts=parts, fs=fs, element=element, shear=shear, shift=shift)


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
    # sheared slab: sliding varies across y (periodic), so there is shear across the y-seam. No exact formula;
    # the reference is the same run shifted by one cell (s1), where the seam sits at an interior node
    for parts in ("serial", "xcut"):
        for shift in (0, 1):
            cs.append(case(f"shear_ny4_periodic_{parts}_s{shift}", ny=4, parts=parts, shear=True, shift=shift))
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
    LOADS_KW = loads_keywords()
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
    if OPTS["loads"] and OPTS["fix"]:
        fix = os.path.join(REPO, "elmer", "FixPeriodicLoads")
        add(f"""  Equation = "FixPeriodicLoads"
  Procedure = "{fix}" "FixPeriodicLoads"
  Loads Variable = String "Flow Solution Loads"
  Target Variable = String "{FIXED}"
""")
    add(f"""  Exec Solver = "After All"
  Equation = "StateWrite"
  Procedure = "{ess}" "StateIO"
  State Mode = String "write"
  State File = File "out.bin"
  Write Coordinates = Logical True
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
OPTS = dict(weights=True, loads=True, fix=True)
FIXED = "Periodic Flow Solution Loads"          # FixPeriodicLoads' output


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
        if OPTS["fix"]:
            names += [f"{FIXED} {k}" for k in (1, 2, 3)]
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
        f.write(PARAMS + f"Wy = {c['ny'] * g['dy']!r}\nshift = {c['shift'] * g['dy']!r}\n")
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
    if c["shear"]:                      # C(y) = Cslip (1 + 0.5 sin(2 pi (y + shift) / W)): shear across the seam
        for k in (2, 3):
            old = f"  Slip Coefficient {k} = Real #Cslip"
            assert old in sif
            sif = sif.replace(old, f"  Slip Coefficient {k} = Variable Coordinate 2\n"
                                   f'    Real Lua "Cslip*(1 + 0.5*math.sin(2*math.pi*(tx[0] + shift)/Wy))"')
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


def load_nodes(c, quiet=False):
    """one record per node (first partition copy) of a finished case, or None (says why)"""
    d = os.path.join(OUT, c["name"])
    if not os.path.isdir(d):
        return None
    st = os.path.join(d, "status.txt")
    status = "?"
    if os.path.exists(st):
        with open(st) as f:
            status = f.read().strip()
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
    raw_n = [f"flow solution loads {k}" for k in (1, 2, 3)]
    fix_n = [f"{FIXED.lower()} {k}" for k in (1, 2, 3)]
    have_fix = all(n in parts[0] for n in fix_n)
    missing = [n for n in raw_n + ["flow solution boundary weights"] + fix_n if n not in parts[0]]
    if missing and not quiet:
        print(f"  {c['name']}: not written: {', '.join(missing)}")
    if any(n not in parts[0] for n in raw_n):
        return None
    g = GEOM
    L, W = g["L"], c["ny"] * g["dy"]
    rows = {}                               # every copy of every node (shared nodes: once per partition)
    for rec in parts:
        gids = rec["coordinate 1"][0]
        X, Y, Z = (rec[f"coordinate {k}"][1] for k in (1, 2, 3))
        get = lambda name: dict(zip(*rec[name])) if name in rec else {}
        Lc = [get(n) for n in raw_n]
        Fc = [get(n) for n in fix_n]
        Wt = get("flow solution boundary weights")
        for i, gid in enumerate(gids):
            r = rows.setdefault(int(gid), dict(x=X[i], y=Y[i], z=Z[i], L=[], F=[], w=[]))
            r["L"].append([Lc[k].get(gid, np.nan) for k in range(3)])
            r["F"].append([Fc[k].get(gid, np.nan) for k in range(3)])
            r["w"].append(Wt.get(gid, np.nan))
    gid = np.array(sorted(rows))
    N = dict(x=np.array([rows[k]["x"] for k in gid]), y=np.array([rows[k]["y"] for k in gid]),
             z=np.array([rows[k]["z"] for k in gid]))
    x, y = N["x"], N["y"]
    N["ncopy"] = np.array([len(rows[k]["L"]) for k in gid])
    N["Lsum"] = np.array([np.nansum(rows[k]["L"], axis=0) for k in gid])     # summed over partition copies
    N["w"] = np.array([rows[k]["w"][0] for k in gid])
    N["raw"] = np.array([rows[k]["L"][0] for k in gid])
    if have_fix:
        N["fix"] = np.array([rows[k]["F"][0] for k in gid])
    # "half": masters divided by their group size, images left as they are (the ElmerStreams fix)
    xe0, xeL = np.abs(x) < 1, np.abs(x - L) < 1
    ye0, yeW = (np.abs(y) < 1, np.abs(y - W) < 1) if c["sides"] == "periodic" else (0 * x > 1, 0 * x > 1)
    image = xeL | yeW
    nmem = np.where(xe0 | xeL, 2, 1) * np.where(ye0 | yeW, 2, 1)
    N["half"] = N["raw"] / np.where(image, 1, nmem)[:, None]

    def spread(key):
        return np.array([np.nanmax(np.ptp(np.array(rows[k][key]), axis=0)) if len(rows[k][key]) > 1 else 0.0
                         for k in gid])
    N["spread_raw"] = spread("L")
    N["spread_fix"] = spread("F") if have_fix else None
    N["H"] = N["z"].max()
    N["A"] = own_area(x, y, L, W, g["dx"], g["dy"])
    N["bed"] = np.abs(N["z"]) < 1e-3
    return N


def variants(N):
    return [v for v in ("raw", "half", "fix") if v in N]


def analyse_case(c):
    N = load_nodes(c)
    if N is None:
        return None
    g = GEOM
    L, W = g["L"], c["ny"] * g["dy"]
    x, y, z, A, bed, H = N["x"], N["y"], N["z"], N["A"], N["bed"], N["H"]
    top = np.abs(z - z.max()) < 1e-3
    # exact bed loads (magnitudes); the sign of Elmer's loads is taken from the data
    ex = dict(x=RHOG * SINA * H * A, z=RHOG * COSA * H * A)
    if np.isnan(N["raw"][bed]).all():
        return None
    sgn = {}
    for k, j in (("x", 0), ("z", 2)):
        v = N["raw"][bed & (A == A.max()), j]
        sgn[k] = (float(np.sign(np.nanmedian(v))) if np.isfinite(v).any() else 1.0) or 1.0
    xe = (np.abs(x) < 1) | (np.abs(x - L) < 1)               # x-periodic ends
    ye = (np.abs(y) < 1) | (np.abs(y - W) < 1)               # the sides
    cls = {"y0": bed & ~xe & (np.abs(y) < 1),                # y = 0: master if y-periodic
           "yW": bed & ~xe & (np.abs(y - W) < 1),            # y = W: image
           "int": bed & ~xe & ~ye,
           "x0": bed & (np.abs(x) < 1) & ~ye,                # x = 0: master of x = L
           "xL": bed & (np.abs(x - L) < 1) & ~ye,
           "cor": bed & xe & ye}                             # corners (master or image in two BCs)
    out = dict(case=c["name"], ny=c["ny"], sides=c["sides"], parts=c["parts"], fs=c["fs"], H=H)
    ref = float(np.median(ex["z"][bed & (A == A.max())]))
    for var in variants(N):
        Lv = N[var]
        for name, m in cls.items():
            if not m.any():
                continue
            for comp, j in (("z", 2), ("x", 0)):
                r1 = sgn[comp] * Lv[m, j] / ex[comp][m]
                out[f"{var}_{name}_{comp}"] = float(np.mean(r1))
                out[f"{var}_{name}_{comp}_sprd"] = float(np.ptp(r1))
            out[f"{var}_{name}_y"] = float(np.mean(Lv[m, 1] / ex["z"][m]))   # cross-flow / exact vertical (0)
        # worst node, vertical load (the along-flow one is small, sin a, and shows the seam force strongly)
        out[f"{var}_maxerr_z"] = float(np.nanmax(np.abs(sgn["z"] * Lv[bed, 2] / ex["z"][bed] - 1)))
        out[f"{var}_maxerr_x"] = float(np.nanmax(np.abs(sgn["x"] * Lv[bed, 0] / ex["x"][bed] - 1)))
        if c["sides"] == "periodic":                          # no walls: the cross-flow load should vanish
            out[f"{var}_maxy"] = float(np.nanmax(np.abs(Lv[bed, 1])) / ref)
        out[f"{var}_top_max"] = float(np.nanmax(np.abs(Lv[top])) / ref)
    for name in ("y0", "yW"):
        m = cls[name]
        if m.any():
            out[f"{name}_w"] = float(np.mean(N["w"][m] / A[m]))          # Elmer's boundary weight / own area
            if (N["ncopy"][m] > 1).any():
                out[f"{name}_z_sum"] = float(np.mean(sgn["z"] * N["Lsum"][m, 2] / ex["z"][m]))
    out["copy_spread_raw"] = float(np.max(N["spread_raw"][bed]) / ref)
    if N["spread_fix"] is not None:
        out["copy_spread_fix"] = float(np.nanmax(N["spread_fix"][bed]) / ref)
    return out


def analyse_shear(c0, c1):
    """sheared slab: B (shifted one cell) at its seam against A one cell in, where it is an interior node.
    B(y) = A(y + dy), so B's seam y = 0 (= W) is A's interior row y = dy, which owns twice the bed area of
    each seam copy: the right load of each seam copy is half of A's interior load there."""
    A_, B_ = load_nodes(c0), load_nodes(c1)
    if A_ is None or B_ is None:
        return None
    g = GEOM
    L, W, dy = g["L"], c0["ny"] * g["dy"], g["dy"]

    def at(N, yv):                    # bed nodes on row y = yv, not the x ends, ordered by x
        m = N["bed"] & (np.abs(N["y"] - yv) < 1) & (np.abs(N["x"]) > 1) & (np.abs(N["x"] - L) > 1)
        o = np.argsort(N["x"][m])
        return {k: N[k][m][o] for k in variants(N)}
    refA = at(A_, dy)["raw"] / 2.0                       # interior: raw is right
    refA_int = at(A_, 2 * dy)["raw"]                     # interior vs interior: the noise floor
    vz = np.abs(refA[:, 2])
    out = dict(case=c1["name"][:-3], parts=c0["parts"],
               shear_tau_over_N=float(np.max(np.abs(refA[:, 0])) / np.max(vz)))
    out["floor_z"] = float(np.max(np.abs(at(B_, dy)["raw"][:, 2] - refA_int[:, 2]) / (2 * vz)))
    for side, yv in (("master", 0.0), ("image", W)):
        Bs = at(B_, yv)
        for var, v in Bs.items():
            for comp, j in (("z", 2), ("x", 0), ("y", 1)):
                out[f"{var}_{side}_{comp}"] = float(np.max(np.abs(v[:, j] - refA[:, j]) / vz))
    return out


def analyse_all(cases):
    plain = [c for c in cases if not c.get("shear")]
    res = [r for r in (analyse_case(c) for c in plain) if r]
    if res:
        keys = list(dict.fromkeys(k for r in res for k in r))
        with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(res)
        print("\nUNIFORM SLAB: nodal bed loads / exact (1 = right), one partition copy. y0 / yW: the sides (y0"
              "\nmaster, yW image when periodic); int: interior; x0 / xL: the x-periodic ends (x0 master); cor:"
              "\ncorners. _z vertical, _x along flow (small: sin a, so the seam force shows strongly), _y cross-flow"
              "\n/ exact vertical (0 = right without walls). maxerr: worst |ratio - 1| over the bed; maxy: worst"
              "\n|cross-flow| / vertical. RAW: Elmer; HALF: masters / group size, images as they are (ElmerStreams);"
              "\nFIXED: FixPeriodicLoads.")
        for var in ("raw", "half", "fix"):
            show = [f"{var}_{n}_{k}" for n in ("y0", "yW", "int", "x0", "xL", "cor") for k in ("z", "x")]
            show += [f"{var}_y0_y", f"{var}_yW_y", f"{var}_maxerr_z", f"{var}_maxerr_x", f"{var}_maxy"]
            show += [f"copy_spread_{var}"]
            show = [k for k in show if k in keys]
            if not show:
                continue
            print(f"\n{var.upper()}")
            hdr = [k.replace(f"{var}_", "").replace(f"_{var}", "") for k in show]
            print(f"{'case':24s} " + " ".join(f"{h:>9s}" for h in hdr))
            for r in res:
                print(f"{r['case']:24s} " + " ".join(f"{r[k]:9.3g}" if isinstance(r.get(k), float) else f"{'-':>9s}"
                                                      for k in show))
        print(f"(all columns in {os.path.join(OUT, 'summary.csv')})")
    shear = {c["name"]: c for c in cases if c.get("shear")}
    sres = [r for r in (analyse_shear(c, shear[c["name"][:-1] + "1"]) for c in shear.values()
                        if c["shift"] == 0 and c["name"][:-1] + "1" in shear) if r]
    if sres:
        print("\nSHEARED SLAB (sliding varies across y): seam loads against the reference (the same flow one cell"
              "\nin, where the seam row is interior). Worst |error| along the seam / the vertical load, for the"
              "\nmaster (y = 0) and image (y = W) copies; components z, x (along flow), y (cross flow)."
              "\nfloor_z: interior against interior (numerical noise); tau/N: along-flow / vertical load there.")
        for var in ("raw", "half", "fix"):
            show = [f"{var}_{s}_{k}" for s in ("master", "image") for k in ("z", "x", "y")]
            show = [k for k in show if k in sres[0]]
            if not show:
                continue
            print(f"\n{var.upper()}")
            hdr = [k.replace(f"{var}_", "") for k in show] + ["floor_z", "tau/N"]
            print(f"{'case':28s} " + " ".join(f"{h:>9s}" for h in hdr))
            for r in sres:
                vals = [r.get(k) for k in show] + [r["floor_z"], r["shear_tau_over_N"]]
                print(f"{r['case']:28s} " + " ".join(f"{v:9.3g}" if isinstance(v, float) else f"{'-':>9s}"
                                                      for v in vals))
    if not res and not sres:
        print("no results yet")


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
    ap.add_argument("--no-fix", action="store_true", help="without FixPeriodicLoads")
    a = ap.parse_args()
    OPTS["weights"] = not a.no_weights
    OPTS["loads"] = not a.no_loads
    OPTS["fix"] = not a.no_fix
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
        if OPTS["fix"] and not os.path.exists(os.path.join(REPO, "elmer", "FixPeriodicLoads.so")):
            raise SystemExit("elmer/FixPeriodicLoads.so missing: `make -C elmer FixPeriodicLoads.so` (or --no-fix)")
        os.makedirs(OUT, exist_ok=True)
        print(f"footprint {GEOM['L'] / 1e3:g} km x ny * {GEOM['dy'] / 1e3:g} km, {GEOM['dx'] / 1e3:g} km cells, "
              f"{GEOM['nz_layers']} layers, H = {GEOM['H']:g} m; results in {OUT}", flush=True)
        for c in cases:
            prepare_mesh(c, a.elmergrid)
            run_case(c, a.mpirun, a.timeout)
    analyse_all(cases)


if __name__ == "__main__":
    main()
