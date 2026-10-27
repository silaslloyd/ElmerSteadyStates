#!/usr/bin/env python3
"""
Periodic grounding-line test (README_gl.md): does everything stay periodic when the grounding line moves?

A marine slab (gl.sif.in): divide at x = 0, fixed calving front at x = L, sides y = 0 / y = W periodic, bed
sloping to the front with a bump across y, Glen ice, linear Weertman sliding through the contact sliding
function (ElmerStreams' grounding line), free top and bottom surfaces, accumulation. Forced in three phases so
the grounding line moves back and then forward: spin-up, retreat (sliding coefficient lowered), advance
(friction back, more accumulation).

Cases (same everything, only the grounding-line treatment differs):
  old   ElmerStreams now: the original contact test (raw Stokes loads; master doubled by Elmer, image one-sided)
        with the hydrostatic loads doubled on the master side (FixFw; here FixPeriodicLoads mode "fold")
  new   SlidCoef_ContactPeriodic: periodic loads, hydrostatic loads and normals in the test, periodic mask
        after it; and the grounded mask made periodic after GroundedSolver (FixPeriodicLoads mode "min")
Both also write a periodic copy of the Stokes loads (FixPeriodicLoads, "folded") for the analysis.
Each case can be run serial or partitioned (ElmerGrid -partition N 1 1 -periodic 0 1 0, as ElmerStreams).

Run inside a batch job (periodic_test/run_gl.cmd):
    python3 periodic_test/run_gl.py                       # both cases, serial, then the analysis
    python3 periodic_test/run_gl.py --only new --parts 4
    python3 periodic_test/run_gl.py --analyse             # the analysis again
Results in periodic_test/out_gl/<case>/: out.bin_t<step>.<pe> every --out-every steps, elmer.log; the
analysis prints, per output time, the grounding line (per row across y) and the largest master/image
mismatch of each field, and writes out_gl/<case>/periodicity.csv.
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out_gl")
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
from make_mesh import write_footprint  # noqa: E402
from ess.statefile import read_state_file  # noqa: E402

# ---------------------------------------------------------------- geometry, physics and forcing (MPa - a - m)
GEOM = dict(L=64e3, nx=64, W=4e3, ny=4, nz_layers=8)
PHASES = dict(t_spin=150.0, t_retreat=300.0, t_end=450.0,      # phase ends (yr)
              f_retreat=0.3, f_advance=1.0,                    # sliding coefficient factors (spin-up: 1)
              a_spin=0.5, a_retreat=0.5, a_advance=1.5)        # accumulation (m/yr)
DT = 1.0
OUT_EVERY = 10
ICE_BODY = 3            # Buoyancy.F90 reads the gravity from Body 3
# Elmer numbers the bottom / top boundaries of an extruded mesh bcoffset + (k - 1) max_body + body (MeshExtrusion):
# here the 4 footprint edges, so the bed is 4 + 3 = 7 and the surface 4 + 2 x 3 = 10
BED_BC, TOP_BC = 4 + ICE_BODY, 4 + 2 * ICE_BODY

PARAMS = """
yearinsec = 31556926.0
rhoi = 917.0e-6 / yearinsec^2              -- MPa a2 m-2
rhow = 1028.0e-6 / yearinsec^2
gravity = 9.81 * yearinsec^2               -- m a-2
Aglen = 14.52                              -- MPa-3 a-1 (4.6e-25 Pa-3 s-1)
eta = (2.0 * Aglen)^(-1.0/3.0)             -- MPa a^(1/3)
Cw = 1.0e-3                                -- MPa a m-1: tau_b = C u_b (grounded)
Wy = {W!r}
r_iw = 917.0 / 1028.0

function bed(x, y)                         -- sloping down to the front, a bump across y (periodic)
  return 300.0 - 0.012 * x + 40.0 * math.sin(2.0 * math.pi * y / Wy)
end
function H0(x)                             -- initial thickness
  return math.max(700.0 - 8.0e-3 * x, 150.0)
end
function zb0(x, y)                         -- grounded on the bed, or floating
  local b = bed(x, y)
  local h = H0(x)
  if b < 0.0 and h * r_iw < -b then return -h * r_iw end
  return b
end
function zs0(x, y)
  return zb0(x, y) + H0(x)
end
function slipfac(t)
  if t <= {t_spin!r} then return 1.0 end
  if t <= {t_retreat!r} then return {f_retreat!r} end
  return {f_advance!r}
end
function acc(t)
  if t <= {t_spin!r} then return {a_spin!r} end
  if t <= {t_retreat!r} then return {a_retreat!r} end
  return {a_advance!r}
end
"""

DIRECT = """  Linear System Solver = Direct
  Linear System Direct Method = MUMPS
  Linear System Convergence Tolerance = 1.0e-12"""

CASES = {
    "old": dict(contact_kw='  Periodic Contact = Logical False\n  Hydrostatic Loads Variable = String "Fw2"',
                fw_fold=True, mask_sync=False),
    "new": dict(contact_kw='  Periodic Contact = Logical True\n  Hydrostatic Loads Variable = String "Fw"',
                fw_fold=False, mask_sync=True),
}
WRITE_VARS = ["Velocity 1", "Velocity 2", "Velocity 3", "Pressure", "Zs", "Zb", "Bedrock", "GroundedMask",
              "Flow Solution Loads 1", "Flow Solution Loads 2", "Flow Solution Loads 3",
              "Periodic Flow Solution Loads 1", "Periodic Flow Solution Loads 2", "Periodic Flow Solution Loads 3",
              "Fwater 1", "Fwater 2", "Fwater 3", "Normal Vector 1", "Normal Vector 2", "Normal Vector 3"]


def case_dir(name, parts):
    return os.path.join(OUT, name if parts == 1 else f"{name}_p{parts}")


def solvers(c):
    lib = os.path.join(REPO, "elmer", "PeriodicGL")
    ess = os.path.join(REPO, "elmer", "SteadyStateIO")
    blocks, act = [], {"bed": [], "top": [], "ice": []}

    def add(text, body):
        k = len(blocks) + 1
        blocks.append(f"Solver {k}\n{text}End\n")
        act[body].append(k)
    mapper = """  Procedure = "StructuredMeshMapper" "StructuredMeshMapper"
  Active Coordinate = Integer 3
  Minimum Height = Real 1.0
  Variable = -nooutput DummyR
  Mesh Velocity Variable = String "dSdt"
  Mesh Update Variable = String "dS"
  Mesh Velocity First Zero = Logical True
  Exported Variable 1 = -dofs 1 "dS"
  Exported Variable 2 = -dofs 1 "dSdt"
  Adjust Surface = Logical True
"""
    grounded = """  Procedure = "ElmerIceSolvers" "GroundedSolver"
  Variable = "GroundedMask"
  Variable DOFs = 1
  Toler = Real 1.0e-3
  Bedrock Variable = String "Bedrock"
  Exported Variable 1 = -dofs 1 Bedrock
"""
    add('  Exec Solver = "Before All"\n  Equation = "GroundedMaskIni"\n' + grounded, "bed")
    add('  Exec Solver = "Before All"\n  Equation = "MapCoordinateInit"\n' + mapper, "ice")
    add('  Exec Solver = "Before Timestep"\n  Equation = "MapCoordinate"\n' + mapper, "ice")
    add("""  Exec Solver = "Before Timestep"
  Equation = "Normal vector"
  Procedure = "ElmerIceSolvers" "ComputeNormalSolver"
  Variable = "Normal Vector"
  Variable DOFs = 3
  ComputeAll = Logical False
  Optimize Bandwidth = Logical False
""", "ice")
    add("""  Exec Solver = "Before Timestep"
  Equation = "Fw"
  Procedure = "ElmerIceSolvers" "GetHydrostaticLoads"
  Variable = Fw[Fwater:3]
  Variable DOFs = 3
""", "bed")
    if c["fw_fold"]:            # ElmerStreams' FixFw: the hydrostatic loads doubled on the master side
        add(f"""  Exec Solver = "Before Timestep"
  Equation = "Fw2 (FixFw)"
  Procedure = "{lib}" "FixPeriodicLoads"
  Source Variable = String "Fw"
  Target Variable = String "Fw2"
  Periodic Mode = String "fold"
""", "bed")
    add(f"""  Equation = "Stokes-Vec"
  Procedure = "IncompressibleNSVec" "IncompressibleNSSolver"
  Div-Curl Discretization = Logical False
  Stokes Flow = Logical True
  Flow Model = Stokes
  Stabilization Method = String "Stabilized"
  Optimize Bandwidth = Logical True
  Nonlinear System Max Iterations = 50
  Nonlinear System Min Iterations = 2
  Nonlinear System Convergence Tolerance = 1.0e-5
  Nonlinear System Newton After Iterations = 100
  Steady State Convergence Tolerance = 1.0e-5
  Calculate Loads = Logical True
  Exported Variable 1 = -dofs 4 Flow Solution Loads
  Exported Variable 2 = -dofs 1 External Pressure
{DIRECT}
""", "ice")
    add(f"""  Equation = "FixPeriodicLoads"
  Procedure = "{lib}" "FixPeriodicLoads"
  Loads Variable = String "Flow Solution Loads"
  Target Variable = String "Periodic Flow Solution Loads"
""", "ice")
    fs = f"""  Variable DOFs = 1
{DIRECT}
  Nonlinear System Max Iterations = 1
  Nonlinear System Convergence Tolerance = 1.0e-6
  Steady State Convergence Tolerance = 1.0e-5
  Stabilization Method = Stabilized
  Apply Dirichlet = Logical True
"""
    add('  Exec Solver = "After Timestep"\n  Equation = "Free Surface Bottom"\n'
        '  Procedure = "FreeSurfaceSolver" "FreeSurfaceSolver"\n  Variable = "Zb"\n'
        '  Exported Variable 1 = -dofs 1 "Zb Residual"\n' + fs, "bed")
    add('  Exec Solver = "After Timestep"\n  Equation = "Free Surface Top"\n'
        '  Procedure = "FreeSurfaceSolver" "FreeSurfaceSolver"\n  Variable = "Zs"\n'
        '  Exported Variable 1 = -dofs 1 "Zs Residual"\n' + fs, "top")
    add('  Exec Solver = "After Timestep"\n  Equation = "GroundedMaskEnd"\n' + grounded, "bed")
    if c["mask_sync"]:          # GroundedSolver's grounding-line pass is one-sided too
        add(f"""  Exec Solver = "After Timestep"
  Equation = "Periodic mask"
  Procedure = "{lib}" "FixPeriodicLoads"
  Source Variable = String "GroundedMask"
  Target Variable = String "GroundedMask"
  Periodic Mode = String "min"
""", "bed")
    wv = "".join(f'  State Variable {i} = String "{n}"\n' for i, n in enumerate(WRITE_VARS, 1))
    writer = f"""  Equation = "{{eq}}"
  Procedure = "{ess}" "StateIO"
  State Mode = String "write"
  State File = File "out.bin"
  Timestep Suffix = Logical True
  Write Coordinates = Logical True
{wv}"""
    add('  Exec Solver = "Before All"\n' + writer.format(eq="StateWrite0"), "ice")
    add(f'  Exec Solver = "After Timestep"\n  Exec Intervals = Integer {OUT_EVERY}\n'
        + writer.format(eq="StateWrite"), "ice")
    return "\n".join(blocks), act


def write_case(name, parts, nsteps, output_level):
    c = CASES[name]
    d = case_dir(name, parts)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    g = GEOM
    with open(os.path.join(d, "params.lua"), "w") as f:
        f.write(PARAMS.format(W=float(g["W"]), **PHASES))
    write_footprint(os.path.join(d, "footprint"), g["L"], g["W"], g["nx"], g["ny"], body=ICE_BODY)
    if parts > 1:
        with open(os.path.join(d, "elmergrid.log"), "w") as f:
            subprocess.run(["ElmerGrid", "2", "2", "footprint", "-partition", str(parts), "1", "1",
                            "-periodic", "0", "1", "0"], cwd=d, check=True, stdout=f, stderr=subprocess.STDOUT)
    text, act = solvers(c)
    subs = {"__MESH_DIR__": ".", "__NZ1__": str(g["nz_layers"] + 1), "__NSTEPS__": str(nsteps),
            "__DT__": repr(DT), "__BED_BC__": str(BED_BC), "__TOP_BC__": str(TOP_BC), "__W__": repr(float(g["W"])), "__OUTPUT_LEVEL__": str(output_level),
            "__BUOY__": os.path.join(REPO, "elmer", "Buoyancy"),
            "__CONTACT_LIB__": os.path.join(REPO, "elmer", "PeriodicGL"),
            "__CONTACT_FN__": "SlidCoef_ContactPeriodic", "__CONTACT_KW__": c["contact_kw"],
            "__SOLVERS__": text}
    for k, b in (("BED", "bed"), ("TOP", "top"), ("ICE", "ice")):
        subs[f"__N{k}__"] = str(len(act[b]))
        subs[f"__{k}__"] = " ".join(map(str, act[b]))
    sif = open(os.path.join(HERE, "gl.sif.in")).read()
    for k, v in subs.items():
        sif = sif.replace(k, v)
    left = re.findall(r"__[A-Z0-9_]+__", sif)
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    open(os.path.join(d, "case.sif"), "w").write(sif)
    open(os.path.join(d, "ELMERSOLVER_STARTINFO"), "w").write("case.sif\n1\n")
    return d


def run_case(name, parts, nsteps, mpirun, output_level):
    d = write_case(name, parts, nsteps, output_level)
    cmd = (mpirun.format(n=parts).split() if parts > 1 else []) + ["ElmerSolver", "case.sif"]
    t0 = time.time()
    with open(os.path.join(d, "elmer.log"), "w") as f:
        ret = subprocess.run(cmd, cwd=d, stdout=f, stderr=subprocess.STDOUT).returncode
    open(os.path.join(d, "status.txt"), "w").write(f"{ret}\n")
    print(f"  {os.path.basename(d):10s} {parts:2d} procs  exit {ret}  {time.time() - t0:7.1f} s", flush=True)
    if ret != 0:
        log = open(os.path.join(d, "elmer.log"), errors="replace").read().splitlines()
        hits = [l for l in log if any(k in l.lower() for k in ("error", "fatal", "signal", "backtrace"))]
        for l in hits[-8:] + ["    ..."] + log[-15:]:
            print("      | " + l[:160])


# ---------------------------------------------------------------- analysis
def read_step(d, step):
    """{name: {gid: value}} merged over the partitions (first copy of shared nodes)"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"out.bin_t{step}.*"))):
        try:
            rec = read_state_file(f)
        except ValueError as e:
            print(f"  {f}: unreadable ({e})")
            return None
        for name, (g, v) in rec.items():
            dd = out.setdefault(name.lower(), {})
            for gi, vi in zip(g.tolist(), v.tolist()):
                dd.setdefault(gi, vi)
    return out


def steps_of(d):
    s = set()
    for f in glob.glob(os.path.join(d, "out.bin_t*.*")):
        m = re.search(r"out\.bin_t(\d+)\.\d+$", f)
        if m:
            s.add(int(m.group(1)))
    return sorted(s)


def analyse_dir(d):
    g = GEOM
    W, L = g["W"], g["L"]
    steps = steps_of(d)
    if not steps:
        print(f"{os.path.basename(d)}: no output yet")
        return
    rows = []
    print(f"\n{os.path.basename(d)}: grounding line (last grounded x, km) per row across y, and the largest "
          f"master (y = 0) / image (y = W) mismatch")
    hdr = (f"{'t':>6s} {'GL rows y=0..W (km)':>32s} {'mask':>5s} {'Zs':>8s} {'Zb':>8s} {'|du|/u':>8s} "
           f"{'loads raw':>9s} {'loads fix':>9s} {'Fw':>8s} {'normal':>8s}")
    print(hdr)
    for st in steps:
        R = read_step(d, st)
        if R is None or "coordinate 1" not in R:
            continue
        gid = np.array(sorted(R["coordinate 1"]))
        X = np.array([R["coordinate 1"][k] for k in gid])
        Y = np.array([R["coordinate 2"][k] for k in gid])
        Z = np.array([R["coordinate 3"][k] for k in gid])

        def field(name):
            dd = R.get(name.lower(), {})
            return np.array([dd.get(k, np.nan) for k in gid])
        gm = field("GroundedMask")
        onbed = np.isfinite(gm)
        if not onbed.any():
            print(f"{st * DT:6.0f}  (no GroundedMask in this output; written: {', '.join(sorted(R))})")
        # pair every node on y = W with the node on y = 0 of the same footprint column and layer, by number:
        # extruded node = footprint node + layer * (footprint nodes); footprint node (i, j) = j (nx+1) + i + 1
        nx, ny = g["nx"], g["ny"]
        pos = {int(k): i for i, k in enumerate(gid)}
        mW = np.abs(Y - W) < 1
        pairs = []
        for i in np.where(mW)[0]:
            j = pos.get(int(gid[i]) - ny * (nx + 1))
            if j is not None and abs(X[j] - X[i]) < 1 and abs(Y[j]) < 1:
                pairs.append((j, i))
        if not pairs:
            print(f"  t-step {st}: no master/image pairs found (node numbering not as expected?)")
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])

        def mism(name, comps=None, scale=None, sel=None):
            names = [name] if comps is None else [f"{name} {k}" for k in comps]
            v = np.stack([field(n) for n in names], 1)
            s = sel if sel is not None else np.ones(len(a), bool)
            if not s.any():
                return np.nan
            dv = np.abs(v[a][s] - v[b][s])
            if np.isnan(dv).all():
                return np.nan
            ref = scale if scale is not None else 1.0
            return float(np.nanmax(dv) / ref)
        u = np.stack([field(f"Velocity {k}") for k in (1, 2, 3)], 1)
        umax = float(np.nanmax(np.abs(u))) or 1.0
        lz = field("Flow Solution Loads 3")
        lref = float(np.nanmax(np.abs(lz[onbed]))) if onbed.any() else 1.0
        fw = field("Fwater 3")
        fref = float(np.nanmax(np.abs(fw[np.isfinite(fw)]))) if np.isfinite(fw).any() else 1.0
        bedpair = onbed[a] & onbed[b]
        r = dict(step=st, t=st * DT,
                 mask=mism("GroundedMask", sel=bedpair), Zs=mism("Zs"), Zb=mism("Zb"),
                 du=mism("Velocity", (1, 2, 3), umax),
                 loads_raw=mism("Flow Solution Loads", (1, 2, 3), lref, bedpair),
                 loads_fix=mism("Periodic Flow Solution Loads", (1, 2, 3), lref, bedpair),
                 Fw=mism("Fwater", (1, 2, 3), fref, bedpair),
                 normal=mism("Normal Vector", (1, 2, 3), 1.0, bedpair))
        # grounding line per row: the largest x with mask >= 0 (grounded or grounding line)
        gl = []
        for yj in np.unique(np.round(Y[onbed], 3)):
            m = onbed & (np.abs(Y - yj) < 1) & (gm > -0.5)
            gl.append(float(X[m].max()) / 1e3 if m.any() else np.nan)
        r["GL"] = gl
        rows.append(r)
        gls = " ".join(f"{v:5.1f}" for v in gl)
        fmt = lambda v: f"{v:8.2g}" if np.isfinite(v) else f"{'-':>8s}"
        print(f"{r['t']:6.0f} {gls:>32s} {r['mask']:5.0f} {fmt(r['Zs'])} {fmt(r['Zb'])} {fmt(r['du'])} "
              f"{fmt(r['loads_raw']):>9s} {fmt(r['loads_fix']):>9s} {fmt(r['Fw'])} {fmt(r['normal'])}")
    if not rows:
        return
    nrow = max(len(r["GL"]) for r in rows)
    for r in rows:
        r["GL"] = r["GL"] + [np.nan] * (nrow - len(r["GL"]))
    with open(os.path.join(d, "periodicity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = ["step", "t", "mask", "Zs", "Zb", "du", "loads_raw", "loads_fix", "Fw", "normal"]
        w.writerow(keys + [f"GL_row{j}" for j in range(nrow)])
        for r in rows:
            w.writerow([r[k] for k in keys] + r["GL"])
    gl0 = np.array([r["GL"][0] if r["GL"] else np.nan for r in rows])
    worst = {k: np.nanmax([r[k] for r in rows]) for k in ("mask", "Zs", "Zb", "du", "loads_fix")}
    if np.isfinite(gl0).any():
        print(f"  grounding line at y = 0: {np.nanmin(gl0):.1f} .. {np.nanmax(gl0):.1f} km "
              f"(start {gl0[0]:.1f}, end {gl0[-1]:.1f})")
    print(f"  master/image mismatch over the run: "
          + ", ".join(f"{k} {v:.2g}" for k, v in worst.items()))
    print("  mask: largest |mask(y=0) - mask(y=W)| (0, 1 or 2); Zs, Zb in m; |du|/u: velocity / max speed;"
          " loads / max vertical load; Fw / max water load; normal: |n(y=0) - n(y=W)|")


def main():
    global OUT_EVERY
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(CASES), default=list(CASES))
    ap.add_argument("--parts", type=int, default=1, help="MPI partitions (1: serial)")
    ap.add_argument("--steps", type=int, default=int(PHASES["t_end"] / DT))
    ap.add_argument("--out-every", type=int, default=OUT_EVERY)
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--mpirun", default=os.environ.get("ESS_MPIRUN", "mpirun -np {n}"))
    ap.add_argument("--output-level", type=int, default=4)
    a = ap.parse_args()
    OUT_EVERY = a.out_every
    if not a.analyse:
        for lib in ("SteadyStateIO.so", "Buoyancy.so", "PeriodicGL.so"):
            if not os.path.exists(os.path.join(REPO, "elmer", lib)):
                raise SystemExit(f"elmer/{lib} missing: make -C elmer && make -C elmer PeriodicGL.so")
        os.makedirs(OUT, exist_ok=True)
        print(f"marine slab {GEOM['L'] / 1e3:g} x {GEOM['W'] / 1e3:g} km, {GEOM['nx']} x {GEOM['ny']} cells, "
              f"{GEOM['nz_layers']} layers; {a.steps} steps of {DT:g} yr; phases {PHASES}", flush=True)
        for name in a.only:
            run_case(name, a.parts, a.steps, a.mpirun, a.output_level)
    for name in a.only:
        for d in sorted(glob.glob(os.path.join(OUT, name)) + glob.glob(os.path.join(OUT, f"{name}_p*"))):
            analyse_dir(d)


if __name__ == "__main__":
    main()
