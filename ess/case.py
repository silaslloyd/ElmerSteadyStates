"""
Self-contained case set-up (`ess setup`): mesh and initial conditions for the thermo-coupled flowline.

Writes into the case directory (default <work>/case):
    footprint.geo, footprint.msh, footprint/      gmsh + ElmerGrid mesh (partitioned for n_procs)
    ics/bed.xy, zs.xy, gm.xy, ctmask.xy           tables along x for the sif initial conditions
    ics/ics.lua                                   initial temperature functions (Lua, used by the sif)

Initial geometry: steady shallow-ice profile for linear (n = 1) viscous ice without sliding, for the
accumulation in ess_utils.lua (margin at 0.8 L, where the flux returns to zero), on the linear bed
b = -dbdx (zscale/xscale) x. Beyond the margin a 30 m 'unglaciated' layer (GroundedMask = 0), as in
ElmerStreams. Initial temperature: conduction only, linear in the ice from Tsurf to the bed,
Tb = min(Tsurf + gthf H / k, T0), geothermal gradient in the bedrock. Temperate bed
(ColdTempMask = +1) only under the ice where Tb reaches T0 - 0.01 K; cold (flux condition) everywhere
else, including beyond the margin. This is only a first guess: spin it
up with `ess run`, or go straight to `ess newton`.

Marine ([case] geometry = "marine"): the ElmerStreams Marine_Term initial state, from its
Initialisation/Dimensional_ICs tables (copied to case/marine_ics): an SIA ice sheet on the bed
-96 m .. -224 m (deepening seaward) with its grounding line where the flux matches the accumulation,
an unconfined shelf beyond it to the calving front at L, and the SIA basal temperature. Grounded
(GroundedMask 1) where Zb is within 0.5 m of the bed, floating (-1) elsewhere; temperate bed only under
grounded ice at T0. Also written: ics/zb.xy (the base).
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess

import numpy as np

log = logging.getLogger("ess")


def eval_lua_parameters(path: str) -> dict:
    """Evaluate a Lua file of `name = expression` lines (as parameters.lua) in Python."""
    v = {}
    with open(path) as f:
        for raw in f:
            line = raw.split("--")[0].strip().rstrip(";").strip()
            if not line or "=" not in line or line.startswith(("function", "if", "print", "local")):
                continue
            k, e = line.split("=", 1)
            k = k.strip()
            if not k.isidentifier():
                continue
            try:
                v[k] = eval(e.replace("^", "**"), {"__builtins__": {}, "math": math}, v)
            except Exception as ex:
                log.warning(f"parameters: cannot evaluate '{line}': {ex}")
    return v


def initial_geometry(P, n=4001, x_end=1.0, edge=30.0):
    """x_end < 1: fixed-margin mode, the domain ends at x_end * L (the margin, 0.8 L), ice everywhere,
    thickness `edge` at the margin."""
    L = P["L"]
    a = P["aDIM"]
    rg = P["rhoi"] * P["gravity"]
    eta = P["viscosity"]
    slope = P["dbdx"] * P["zscale"] / P["xscale"]
    xm = 0.8 * L
    x = np.linspace(0, min(x_end, 1.0) * L, n)
    bed = -slope * x
    # flux of the accumulation (x - 0.4 L) 2a / (-0.4 L): q = 2a (x - x^2 / (0.8 L)), zero at 0 and 0.8 L
    q = lambda s: 2 * a * (s - s * s / (0.8 * L))
    bx = -slope
    # u = H^4:  du/dx = -12 eta q / (rho g) - 4 H^3 b_x, integrated from the margin to the divide
    xs = np.linspace(xm, 0, 40001)
    h = xs[1] - xs[0]
    u = np.zeros(len(xs))
    u[0] = (edge if x_end < 1.0 else 30.0) ** 4

    def f(s, uu):
        return -12 * eta * q(s) / rg - 4 * max(uu, 0.0) ** 0.75 * bx

    for i in range(len(xs) - 1):
        s, uu = xs[i], u[i]
        k1 = f(s, uu)
        k2 = f(s + h / 2, uu + h / 2 * k1)
        k3 = f(s + h / 2, uu + h / 2 * k2)
        k4 = f(s + h, uu + h * k3)
        u[i + 1] = uu + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    H_ice = np.interp(x, xs[::-1], np.maximum(u[::-1], 0) ** 0.25, right=0.0)
    if x_end < 1.0:
        H = np.maximum(H_ice, edge)              # ice everywhere; the end is the margin
        gm = np.ones_like(x)
    else:
        gm = (H_ice > 150).astype(float)
        H = np.where(gm > 0, H_ice, 30.0)
    zs = bed + H
    Ts = P["Tsurf"]
    Tb = np.minimum(Ts + P["gthf"] / P["heatconductivity"] * H, P["T0"])
    ct = np.where((gm > 0) & (Tb >= P["T0"] - 0.01), 1.0, -1.0)     # temperate only under ice
    return dict(x=x, bed=bed, zs=zs, H=H, gm=gm, ct=ct, Tb=Tb, Ts_surf=np.full_like(x, Ts))


def marine_geometry(P, icdir, n=8001, rho_tol=0.5):
    """ElmerStreams Marine_Term initial state from its dimensional tables (x in m, values in m / K)."""
    L = P["L"]
    x = np.linspace(0, L, n)
    tab = lambda f: np.loadtxt(os.path.join(icdir, f))
    at = lambda f: np.interp(x, *tab(f).T)
    bed, zs, zb, Tb = at("bed_dim.xy"), at("zs_dim.xy"), at("zb_dim.xy"), at("Tb_dim.xy")
    zb = np.maximum(zb, bed)
    H = zs - zb
    gm = np.where(zb - bed < rho_tol, 1.0, -1.0)
    Tb = np.where(gm > 0, np.minimum(Tb, P["T0"]), P["T0"])
    ct = np.where((gm > 0) & (Tb >= P["T0"] - 0.01), 1.0, -1.0)
    return dict(x=x, bed=bed, zs=zs, zb=zb, H=H, gm=gm, ct=ct, Tb=Tb, Ts_surf=np.full_like(x, P["Tsurf"]))


def _lua_array(name, a):
    return f"{name} = {{" + ", ".join(f"{v:.10g}" for v in a) + "}\n"


ICS_LUA = """
-- interpolation on the tables above (x ascending)
function ics_interp(ys, x)
  local n = #ics_x
  if x <= ics_x[1] then return ys[1] end
  if x >= ics_x[n] then return ys[n] end
  local lo, hi = 1, n
  while hi - lo > 1 do
    local mid = math.floor((lo + hi) / 2)
    if ics_x[mid] <= x then lo = mid else hi = mid end
  end
  local t = (x - ics_x[lo]) / (ics_x[hi] - ics_x[lo])
  return ys[lo] + t * (ys[hi] - ys[lo])
end

-- initial temperature; z is the extruded (unmapped) coordinate: bedrock 0..0.5, ice 0.5..1
function Tinit_ice(x, z)
  local zn = (z - 0.5) / 0.5
  if zn < 0 then zn = 0 end
  if zn > 1 then zn = 1 end
  local Tb = ics_interp(ics_Tb, x)
  local Ts = ics_interp(ics_Ts, x)
  return Tb + (Ts - Tb) * zn
end

function Tinit_rock(x, z)
  local d = (0.5 - z) / 0.5 * 10 * zscale
  if d < 0 then d = 0 end
  return ics_interp(ics_Tb, x) + gthf / heatconductivityrock * d
end
"""


def write_ics(g, icdir):
    os.makedirs(icdir, exist_ok=True)
    for name, key in [("bed", "bed"), ("zs", "zs"), ("gm", "gm"), ("ctmask", "ct")] + ([("zb", "zb")] if "zb" in g else []):
        np.savetxt(os.path.join(icdir, f"{name}.xy"), np.column_stack([g["x"], g[key]]), fmt="%.10g")
    # coarser tables for Lua
    k = max(1, len(g["x"]) // 800)
    with open(os.path.join(icdir, "ics.lua"), "w") as f:
        f.write("-- generated by ess setup\n")
        f.write(_lua_array("ics_x", g["x"][::k]))
        f.write(_lua_array("ics_Tb", g["Tb"][::k]))
        f.write(_lua_array("ics_Ts", g["Ts_surf"][::k]))
        f.write(ICS_LUA)


def _run(cmd, cwd):
    log.info("setup: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise SystemExit(f"setup: command failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout[-3000:]}")
    return r.stdout


def setup_case(cfg, mesh=True):
    case = cfg["paths"]["case_dir"]
    os.makedirs(case, exist_ok=True)
    params = cfg["_path"](cfg.get("case", {}).get("parameters", "case/parameters.lua"))
    shutil.copy(params, os.path.join(case, "parameters.lua"))
    P = eval_lua_parameters(params)
    pname = cfg["parameter"]["name"]
    if pname in P and "value" in cfg["parameter"]:
        P[pname] = float(cfg["parameter"]["value"])
    c = cfg.get("case", {})
    x_end = float(c.get("x_end", 1.0))
    marine = c.get("geometry", "land") == "marine"
    if marine:
        g = marine_geometry(P, cfg["_path"](c.get("marine_ics", "case/marine_ics")))
    else:
        g = initial_geometry(P, x_end=x_end, edge=float(c.get("edge_thickness", 30.0)))
    write_ics(g, os.path.join(case, "ics"))
    xm = g["x"][g["gm"] > 0].max()
    log.info(f"setup: L = {P['L'] / 1e3:.1f} km, {'grounding line' if marine else 'margin'} at {xm / 1e3:.1f} km, "
             f"max thickness {g['H'].max():.0f} m, "
             + (f"front thickness {g['H'][-1]:.0f} m, " if marine else "")
             + f"bed temperature {g['Tb'][g['gm'] > 0].min():.2f}..{g['Tb'][g['gm'] > 0].max():.2f} K, "
             f"cold-bed fraction {np.mean(g['ct'][g['gm'] > 0] < 0):.2f}")
    P["GLx"] = float(xm)
    if "gl_from_state" in c:                  # refine around the grounding line of this state (another mesh is fine)
        from .masks import _on
        from .statefile import find_name, load_state
        st = load_state(cfg["_path"](c["gl_from_state"]))
        g_, m_ = st.frozen[find_name(st.frozen, "GroundedMask")]
        x_ = _on(*st.diag[find_name(st.diag, "coordinate 1")], g_)
        P["GLx"] = float(x_[m_ > -0.5].max())
        log.info(f"setup: mesh refined around the grounding line of {c['gl_from_state']}: x = {P['GLx'] / 1e3:.2f} km")
    elif "gl_x_km" in c:                      # refine around this grounding line instead of the initial one
        P["GLx"] = float(c["gl_x_km"]) * 1e3
        log.info(f"setup: mesh refined around x = {P['GLx'] / 1e3:.2f} km ([case] gl_x_km)")
    P["FineRes"] = float(c.get("fine_res", P.get("MarginRes", 1000.0)))       # flowline_marine_gl.geo.in
    P["FineHalf"] = float(c.get("fine_half_width", 10e3))
    if not mesh:
        return g
    geo = open(cfg["_path"](cfg.get("case", {}).get("geo", "case/flowline.geo.in"))).read()
    P["Lend"] = min(x_end, 1.0) * P["L"]
    if "width" in c:                          # m, width of the strip across the flow
        P["Wmesh"] = float(c["width"])
    ny = int(c.get("y_elements", 1))          # elements across the flow
    geo = geo.replace("{YElements}", str(ny))
    for k in ("L", "Lend", "Wmesh", "BoundaryRes", "MarginRes", "GLx", "FineRes", "FineHalf"):
        if "{" + k + "}" in geo:
            geo = geo.replace("{" + k + "}", repr(float(P[k])))
    if ny > 1:
        log.info(f"setup: {ny} elements across a width of {P['Wmesh'] / 1e3:g} km")
    with open(os.path.join(case, "footprint.geo"), "w") as f:
        f.write(geo)
    _run([c.get("gmsh", "gmsh"), "footprint.geo", "-1", "-2", "-format", "msh2", "-o", "footprint.msh"], case)
    n = (int(os.environ.get("ESS_NPROCS", 0)) or int(cfg["elmer"].get("n_procs", 0))
         or int(os.environ.get("SLURM_NTASKS", "1")))
    eg = c.get("elmergrid", "ElmerGrid")
    out = os.path.join(case, cfg["elmer"].get("mesh_db", "footprint"))
    if os.path.isdir(out):
        shutil.rmtree(out)
    cmd = [eg, "14", "2", "footprint.msh", "-autoclean", "-out", cfg["elmer"].get("mesh_db", "footprint")]
    periodic = bool(c.get("periodic_sides", False))
    if periodic:                              # side 2 the periodic image of side 1 (ElmerStreams flowline.cmd periodic)
        cmd += ["-periodic", "0", "1", "0"]
    if n > 1:
        part = c.get("partition")              # [px, py]: px x py boxes (px * py = n), e.g. [64, 4] (stream.cmd)
        if part:
            px, py = (int(v) for v in part)
            if px * py != n:
                raise SystemExit(f"[case] partition {px} x {py} != {n} processes")
            cmd += ["-partition", str(px), str(py), "1"]
        else:
            cmd += ["-partition", str(n), "1", "1"] if periodic else ["-metiskway", str(n)]
    _run(cmd, case)
    log.info(f"setup: mesh in {out}" + (f" (partitioned for {n})" if n > 1 else ""))
    return g
