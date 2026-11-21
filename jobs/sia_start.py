#!/usr/bin/env python3
"""A Newton start from a dimensionless SIA steady profile (plots/data/sia_profile.npz: x, bed, h = thickness, Tb = basal
temperature, 0 = melting; plots/plot_section.py --sia draws it), scaled as plot_section's "temperate fit":

    x scale X = x_gl / x_gl,SIA        (the SIA's flotation point on the run's grounding line)
    surface Zs = Z (bed_SIA + h) + c   (Z, c by least squares on the run's surface over the SIA's temperate range)

against a converged sweep state SRC (default: the refined sweep's newest point). Then:

    1. the mesh: the base config's res_points with a FINE m band around the SIA's cold/temperate transition (the new
       patch head; d = x_gl - head), config <work>/refined/s<head>.toml, mesh <work>/s<head>/ (as jobs/jump_refined.py);
    2. ess setup, ess init, ess transfer --from SRC (the shelf, the bedrock, the masks), jobs/shelf_break.py;
    3. the grounded surface replaced by the SIA's (clipped to Zb + 1 m), the patch for d -> <work>/sia/s<head>/start_sia.npz;
    4. steady heat on that fixed geometry: ess run --heat-only (Stokes and heat, no free surface), HEAT_STEPS x HEAT_DT
       yr, the grounding line frozen, no remasking -> <work>/sia/s<head>/start_heat.npz;
    5. Newton with the contact grounding-line rounds (ess sweep at that one d) -> <work>/sia/s<head>/sweep/.
Rerunning resumes (each step's output kept).

    python3 jobs/sia_start.py CONFIG [--from-state SRC.npz] [--sia plots/data/sia_profile.npz] [--pad-km 10]
                              [--fine 250] [--heat-dt 1e6] [--heat-steps 2] [--gl-rounds 6] [--no-newton]
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sweep_refined import banded_points, converged, write_config  # noqa: E402
from ess.config import load_config  # noqa: E402
from ess.spectrum import gl_position, patch_mask  # noqa: E402
from ess.statefile import find_name, load_state, save_state  # noqa: E402

RHO_I, RHO_W = 920.0, 1003.27            # case/parameters.lua


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} sia: {msg}", flush=True)


def run(cmd, env=None):
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def column_profile(s, name):
    """a nodal block's value per column (x in m, sorted; the columns' nodes averaged)"""
    gx, X = s.diag[find_name(s.diag, "coordinate 1")]
    blk = next(b for b in s.layout.blocks if b.name.lower() == name)
    v = s.x[s.layout.slices()[blk.name]]
    x = X[np.searchsorted(gx, blk.gids)]
    xr = np.round(x, 3)
    u, inv = np.unique(xr, return_inverse=True)
    return u, np.bincount(inv, v) / np.bincount(inv)


def bed_profile(s):
    gx, X = s.diag[find_name(s.diag, "coordinate 1")]
    g, v = s.frozen[find_name(s.frozen, "bedrock")]
    x = np.round(X[np.searchsorted(gx, g)], 3)
    u, k = np.unique(x, return_index=True)
    return u, v[k]


def sia_fit(src, path):
    """the temperate fit: (X m, Z m, c m, x_t m (the SIA's transition on the run), the SIA's x, bed, h; the fit's rms)"""
    z = np.load(path)
    xd, bd, hd, tb = z["x"], z["bed"], z["h"], z["Tb"]
    fl = hd * RHO_I / RHO_W < -bd
    i = int(np.argmax(fl)) if fl.any() else len(xd) - 1
    xd, bd, hd, tb = xd[:i + 1], bd[:i + 1], hd[:i + 1], tb[:i + 1]
    warm = tb >= -1e-9
    if not warm.any() or warm[0]:
        raise SystemExit(f"{path}: no single cold/temperate transition in Tb")
    xt = float(xd[np.argmax(warm)])
    xgl = gl_position(src.frozen, src.diag)
    X = xgl / float(xd[-1])
    xs, zs = column_profile(src, "zs")
    mt = xd >= xt
    A = np.c_[bd[mt] + hd[mt], np.ones(int(mt.sum()))]
    (Z, c), *_ = np.linalg.lstsq(A, np.interp(xd[mt] * X, xs, zs), rcond=None)
    rms = float(np.sqrt(np.mean((A @ [Z, c] - np.interp(xd[mt] * X, xs, zs)) ** 2)))
    return float(X), float(Z), float(c), xt * X, (xd, bd, hd), rms, xgl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--from-state")
    ap.add_argument("--sia", default=os.path.join(ROOT, "plots", "data", "sia_profile.npz"))
    ap.add_argument("--pad-km", type=float, default=10.0)
    ap.add_argument("--grade-km", type=float, default=10.0)
    ap.add_argument("--fine", type=float, default=250.0)
    ap.add_argument("--heat-dt", type=float, default=1e6)
    ap.add_argument("--heat-steps", type=int, default=2)
    ap.add_argument("--gl-rounds", type=int, default=6)
    ap.add_argument("--no-newton", action="store_true")
    ap.add_argument("--shelf-break", default="--strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 --width-km 10")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work = cfg["_work"]
    c = cfg["case"]
    base = c["res_points"]
    front_km = float(c.get("front_km", base[-1][0]))
    if a.from_state:
        srcf = a.from_state
    else:
        pts = [p for f in sorted(glob.glob(os.path.join(work, "sweep_refined", "seg_*"))) for p in converged(f)]
        if not pts:
            raise SystemExit("no converged point of the refined sweep")
        srcf = min(pts)[1]                      # the smallest d: the newest point
    src = load_state(srcf)
    X, Z, cz, xt, (xd, bd, hd), rms, xgl = sia_fit(src, a.sia)
    head = xt / 1e3
    d = round((xgl / 1e3 - head) * 10) / 10
    tag = f"s{int(round(head)):04d}"
    log(f"from {os.path.relpath(srcf, ROOT)} (grounding line {xgl / 1e3:.2f} km); SIA temperate fit: x scale "
        f"{X / 1e3:.1f} km, z scale {Z:.0f} m, translated {cz:+.0f} m (rms surface misfit {rms:.1f} m over the "
        f"temperate range); the SIA's transition at {head:.2f} km: d = {d:g} km ({tag})")
    # 1-2. the mesh and the transferred start
    pts = banded_points(base, head - a.pad_km, head + a.pad_km, a.fine, a.grade_km, front_km)
    cfgf = os.path.join(work, "refined", f"{tag}.toml")
    mdir = os.path.join(work, tag)
    os.makedirs(os.path.dirname(cfgf), exist_ok=True)
    write_config(a.config, cfgf, pts, "{work}/" + tag)
    log(f"mesh: band {a.fine:g} m over {head - a.pad_km:.1f} .. {head + a.pad_km:.1f} km; res_points {pts}")
    py = sys.executable
    st0, st1, st2 = (os.path.join(mdir, f) for f in ("state0.npz", "start0.npz", "start.npz"))
    if not os.path.exists(st0):
        run([py, "-u", "-m", "ess", "setup", cfgf])
        run([py, "-u", "-m", "ess", "init", cfgf, "--out", st0])
    if not os.path.exists(st1):
        run([py, "-u", "-m", "ess", "transfer", cfgf, "--state", st0, "--from", srcf, "--out", st1])
    if not os.path.exists(st2):
        run([py, "jobs/shelf_break.py", st1, st2] + a.shelf_break.split() + ["--force"])
    sdir = os.path.join(work, "sia", tag)
    os.makedirs(sdir, exist_ok=True)
    # 3. the SIA surface on the grounded columns, the patch for d
    st3 = os.path.join(sdir, "start_sia.npz")
    if not os.path.exists(st3):
        s = load_state(st2)
        gx, Xc = s.diag[find_name(s.diag, "coordinate 1")]
        sl = s.layout.slices()
        bz = next(b for b in s.layout.blocks if b.name.lower() == "zs")
        xz = Xc[np.searchsorted(gx, bz.gids)]
        zs_sia = cz + Z * np.interp(xz / X, xd, bd + hd)
        grounded = xz <= xgl + 1.0              # the columns up to the grounding line (the source's, as the mesh's)
        zb_on_zs = np.interp(xz, *column_profile(s, "zb"))
        x = s.x.copy()
        old = x[sl[bz.name]].copy()
        new = np.where(grounded, np.maximum(zs_sia, zb_on_zs + 1.0), old)
        x[sl[bz.name]] = new
        dz = new - old
        log(f"surface: {int(grounded.sum())} grounded Zs values set to the SIA's (change {dz[grounded].min():+.0f} .. "
            f"{dz[grounded].max():+.0f} m, rms {np.sqrt(np.mean(dz[grounded] ** 2)):.0f} m); "
            f"{int((~grounded).sum())} floating kept")
        frozen = dict(s.frozen)
        kct = find_name(frozen, "ColdTempMask")
        g_ct, ct, xg = patch_mask(frozen, s.diag, d * 1e3)
        frozen[kct] = (g_ct, ct)
        log(f"patch: d = {d:g} km, {int(np.sum(ct > 0))} temperate bed nodes (incl. floating)")
        save_state(st3, s.layout, x, s.p, frozen, s.diag, sweep_d=d * 1e3, x_gl=xg, source=srcf,
                   sia_scale_x=X, sia_scale_z=Z, sia_shift=cz)
    # 4. steady heat on the fixed geometry
    st4 = os.path.join(sdir, "start_heat.npz")
    if not os.path.exists(st4):
        env = dict(os.environ, ESS_SEA_SPRING_DT=os.environ.get("ESS_SEA_SPRING_DT", "50"))   # the map's spring
        run([py, "-u", "-m", "ess", "run", cfgf, "--state", st3, "--heat-only", "--dt", f"{a.heat_dt:g}",
             "--steps", str(a.heat_steps), "--gl", "frozen", "--no-remask", "--out", st4], env=env)
        h = load_state(st4)
        if "sweep_d" not in h.meta:                 # ess sweep's resume needs no meta; keep d for the plots
            save_state(st4, h.layout, h.x, h.p, h.frozen, h.diag, **h.meta, sweep_d=d * 1e3)
    if a.no_newton:
        return
    # 5. Newton with the contact rounds at that d
    run([py, "-u", "-m", "ess", "sweep", cfgf, "--state", st4, "--d-start-km", f"{d:g}", "--d-end-km", f"{d:g}",
         "--gl-rounds", str(a.gl_rounds), "--out", os.path.join(sdir, "sweep")])
    log("done")


if __name__ == "__main__":
    main()
