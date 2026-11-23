#!/usr/bin/env python3
"""The boundary of HYDROLOGICALLY COMPATIBLE steady states along the patch sweep: the steady water sheet on the
temperate patch (plots/sweep_hydrology.py: the CS sheet integrated upstream from the grounding line) must not run dry
before the patch head. Compatible ("ok") for short patches (head far inland, e.g. 700 km), "dry" for long ones
(head < 600 km); this finds where it switches, by bisection in d, and converges the states either side:

    1. the converged points of the given sweep folders, each one's sheet status (ok / dry / flood); the first switch
       ok -> dry with increasing d between NEIGHBOURING points (gap <= --max-gap-km). Until one exists (the sweeps
       still running): wait (--wait-min) and look again.
    2. one mesh for the bracket: --fine m over the heads of [d_ok, d_dry] +- --pad-km (config <work>/refined/<tag>.toml,
       mesh <work>/<tag>/, tag z<head>), Newton's map --map-steps; both ends carried onto it and re-converged there.
    3. bisection: Newton at the middle d (rounded to 100 m, the sweep's resolution) from the nearer end's converged
       state; its sheet status replaces that end; until d_dry - d_ok <= --resolution-km (default 0.3: about one
       250 m node of the head).
Output <work>/hydro_bisect/<tag>/: d<XXXXX>/ (each Newton: sweep.csv, points/state_XXXXX.npz), bisect.csv (every d
solved and its status, h at the head, q at the grounding line), and the final bracket in the log. Resumes (every
converged d kept).

    python3 jobs/hydro_bisect.py CONFIG ROOT [ROOT2 ...] [--map-steps 10] [--fine 250] [--pad-km 10]
                                 [--max-gap-km 25.1] [--resolution-km 0.3] [--wait-min 15] [--wait-max-h 12]
ROOT: a sweep folder (seg_*/sweep.csv) or one sweep (sweep.csv), as plots/sweep_rates.py.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from sweep_refined import banded_points, last_converged, write_config  # noqa: E402
from ess.case import eval_lua_parameters  # noqa: E402
from ess.config import load_config  # noqa: E402
from ess.spectrum import _bed, basal_temperature, cs_sheet_params, gl_position, patch_hydrology_cs  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} hydro-bisect: {msg}", flush=True)


def run(cmd):
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def points(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "seg_*", "sweep.csv"))) or [os.path.join(root, "sweep.csv")]:
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            if str(r.get("converged")) != "1":
                continue
            d = round(float(r["d_km"]), 1)
            p = os.path.join(os.path.dirname(f), "points", f"state_{int(round(d * 10)):05d}.npz")
            if os.path.exists(p):
                out[d] = p
    return out


class Sheet:
    """the steady sheet's status of a converged state (as plots/sweep_hydrology.py)"""

    def __init__(self, cfg):
        self.hp = cs_sheet_params(cfg)
        pf = os.path.join(ROOT, cfg.get("case", {}).get("parameters") or "case/parameters_fine2x.lua")
        self.P = eval_lua_parameters(pf)
        self.cache = {}

    def __call__(self, path):
        if path in self.cache:
            return self.cache[path]
        s = load_state(path)
        g, xpos, grounded = _bed(s.frozen, s.diag)
        ct = s.frozen[find_name(s.frozen, "ColdTempMask")]
        ctv = dict(zip(ct[0].tolist(), ct[1]))
        temp = np.array([ctv.get(k, -1.0) > 0 for k in g.tolist()])
        if not (grounded & temp).any():
            r = ("no-patch", {})
        else:
            xg = float(xpos[grounded].max())
            d = xg - float(xpos[grounded & temp].min())
            ok, st, row, _ = patch_hydrology_cs(s.diag, s.layout, s.x, s.frozen, xg, d, dict(self.hp),
                                                rhoi=self.P["rhoi"], Lw=self.P["Lw"], rhow=self.P["rhow"],
                                                gravity=self.P["gravity"], dx_bin=250.0)
            r = (st, row)
        self.cache[path] = r
        return r


class Thermal:
    """thermal consistency of a converged state (as the sweep's n_cold_above_Tm): "ok" when no grounded bed node held
    cold is above T0 + tol_T, else "hot" (the bed above the head wants to be temperate: the patch too short)"""

    def __init__(self, tol_T=0.0):
        self.tol_T = tol_T
        self.cache = {}

    def __call__(self, path):
        if path in self.cache:
            return self.cache[path]
        s = load_state(path)
        g, xpos, grounded = _bed(s.frozen, s.diag)
        ct = s.frozen[find_name(s.frozen, "ColdTempMask")]
        ctv = dict(zip(ct[0].tolist(), ct[1]))
        cold = grounded & np.array([ctv.get(k, -1.0) < 0 for k in g.tolist()])
        Tb = basal_temperature(s.layout, s.x, g)
        hot = cold & (Tb > 273.15 + self.tol_T)
        row = dict(n_cold_above_Tm=int(hot.sum()), Tb_cold_max=float(np.nanmax(Tb[cold])) if cold.any() else np.nan,
                   x_hot_km=float(xpos[hot].min() / 1e3) if hot.any() else np.nan)
        r = ("hot" if hot.any() else "ok", row)
        self.cache[path] = r
        return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--map-steps", type=int, default=10)
    ap.add_argument("--fine", type=float, default=250.0)
    ap.add_argument("--pad-km", type=float, default=10.0)
    ap.add_argument("--grade-km", type=float, default=10.0)
    ap.add_argument("--max-gap-km", type=float, default=25.1)
    ap.add_argument("--resolution-km", type=float, default=0.3)
    ap.add_argument("--wait-min", type=float, default=15.0)
    ap.add_argument("--wait-max-h", type=float, default=12.0)
    ap.add_argument("--gl-rounds", type=int, default=6)
    ap.add_argument("--criterion", choices=("hydro", "thermal"), default="hydro",
                    help="hydro: the sheet ok / dry; thermal: no cold bed node above T0 (ok) / some (hot)")
    ap.add_argument("--list", action="store_true", help="print the points' sheet status and the bracket, then stop")
    ap.add_argument("--shelf-break", default="--strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 --width-km 10")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work = cfg["_work"]
    base = cfg["case"]["res_points"]
    front_km = float(cfg["case"].get("front_km", base[-1][0]))
    sheet = Sheet(cfg) if a.criterion == "hydro" else Thermal()
    bad = "dry" if a.criterion == "hydro" else "hot"
    # 1. the bracket from the sweeps' converged points (wait for the sweeps if there is none yet)
    t0 = time.time()
    while True:
        pts = {}
        for r in a.roots:
            pts.update(points(r))
        ds = sorted(pts)
        stat = {d: sheet(pts[d])[0] for d in ds}
        log(f"{a.criterion} status of the converged points: " + ", ".join(f"{d:g} {stat[d]}" for d in ds))
        br = None
        for d1, d2 in zip(ds, ds[1:]):              # neighbours, one ok and one bad (either order in d)
            if {stat[d1], stat[d2]} == {"ok", bad} and d2 - d1 <= a.max_gap_km:
                br = (d1, d2) if stat[d1] == "ok" else (d2, d1)
                break
        if br or a.list:
            break
        if time.time() - t0 > a.wait_max_h * 3600:
            raise SystemExit(f"no ok / {bad} switch between neighbouring converged points: stopping")
        log(f"no ok / {bad} switch between neighbours (gap <= {a.max_gap_km:g} km) yet: waiting {a.wait_min:g} min")
        time.sleep(a.wait_min * 60)
    if a.list:
        log(f"bracket: {br}" if br else f"no ok / {bad} switch between neighbours")
        return
    d_ok, d_dry = br                                   # (d_dry: the bad end, dry or hot)
    s = load_state(pts[d_ok])
    xgl = gl_position(s.frozen, s.diag) / 1e3
    hlo, hhi = xgl - max(d_ok, d_dry), xgl - min(d_ok, d_dry)
    tag = f"{'z' if a.criterion == 'hydro' else 'y'}{int(round(hhi)):04d}"
    log(f"bracket: d {d_ok:g} km ok (head {xgl - d_ok:.2f} km) .. d {d_dry:g} km {bad} (head {xgl - d_dry:.2f} km); "
        f"mesh {tag}")
    # 2. the bracket's mesh, both ends on it
    pts_m = banded_points(base, hlo - a.pad_km, hhi + a.pad_km, a.fine, a.grade_km, front_km)
    cfgf = os.path.join(work, "refined", f"{tag}.toml")
    os.makedirs(os.path.dirname(cfgf), exist_ok=True)
    write_config(a.config, cfgf, pts_m, "{work}/" + tag, None, a.map_steps)
    mdir = os.path.join(work, tag)
    bdir = os.path.join(work, "hydro_bisect" if a.criterion == "hydro" else "thermal_bisect", tag)
    os.makedirs(bdir, exist_ok=True)
    py = sys.executable
    st0 = os.path.join(mdir, "state0.npz")
    if not os.path.exists(st0):
        if run([py, "-u", "-m", "ess", "setup", cfgf]) or run([py, "-u", "-m", "ess", "init", cfgf, "--out", st0]):
            raise SystemExit("mesh setup failed")
    rows = {}
    csvf = os.path.join(bdir, "bisect.csv")

    def solve(d, start):
        """Newton at d on the bracket's mesh from start (a state on it); the converged state or None"""
        out = os.path.join(bdir, f"d{int(round(d * 10)):05d}")
        got, _ = last_converged(out)
        if got is None:
            run([py, "-u", "-m", "ess", "sweep", cfgf, "--state", start, "--d-start-km", f"{d:g}", "--d-end-km",
                 f"{d:g}", "--gl-rounds", str(a.gl_rounds), "--out", out])
            got, _ = last_converged(out)
        if got is None:
            return None
        st, row = sheet(got[1])
        rows[d] = (st, row, got[1])
        with open(csvf, "w") as f:
            keys = ("h_head", "h_min", "q_gl") if a.criterion == "hydro" else ("n_cold_above_Tm", "Tb_cold_max", "x_hot_km")
            f.write("d_km,head_km,status," + ",".join(keys) + ",state\n")
            for dd in sorted(rows):
                s_, r_, p_ = rows[dd]
                f.write(f"{dd:g},{xgl - dd:.2f},{s_}," + ",".join(str(r_.get(k, '')) for k in keys)
                        + f",{os.path.relpath(p_, ROOT)}\n")
        log(f"d {d:g} km (head {xgl - d:.2f} km): converged, {a.criterion} {st}"
            + (f" ({row['n_cold_above_Tm']} cold nodes above T0, warmest {row['Tb_cold_max'] - 273.15:+.3f} K)"
               if a.criterion == "thermal" else ""))
        return got[1]

    ends = {}
    for d in (d_ok, d_dry):
        st1, st2 = (os.path.join(mdir, f"{f}_{int(round(d * 10)):05d}.npz") for f in ("start0", "start"))
        if not os.path.exists(st1):
            if run([py, "-u", "-m", "ess", "transfer", cfgf, "--state", st0, "--from", pts[d], "--out", st1]):
                raise SystemExit("transfer failed")
        if not os.path.exists(st2):
            if run([py, "jobs/shelf_break.py", st1, st2] + a.shelf_break.split() + ["--force"]):
                raise SystemExit("shelf break failed")
        p = solve(d, st2)
        if p is None:
            raise SystemExit(f"d {d:g} km did not re-converge on the bracket's mesh")
        ends[d] = p
    lo, hi = min(d_ok, d_dry), max(d_ok, d_dry)
    if {rows[lo][0], rows[hi][0]} != {"ok", bad}:
        log(f"on the bracket's mesh the ends are {rows[lo][0]} / {rows[hi][0]}: the switch moved; bisecting what is "
            "there anyway" if rows[lo][0] != rows[hi][0] else "no switch on the bracket's mesh: stopping")
        if rows[lo][0] == rows[hi][0]:
            sys.exit(1)
    st_lo = rows[lo][0]
    # 3. bisection
    while hi - lo > a.resolution_km + 1e-9:
        mid = round((lo + hi) / 2.0, 1)
        if mid in (lo, hi):
            break
        start = ends[lo] if mid - lo <= hi - mid else ends[hi]
        p = solve(mid, start)
        if p is None:
            log(f"d {mid:g} km did not converge: stopping with the bracket {lo:g} .. {hi:g} km")
            sys.exit(1)
        if rows[mid][0] == st_lo:
            lo, ends[lo] = mid, p
        else:
            hi, ends[hi] = mid, p
        log(f"bracket now d {lo:g} ({rows[lo][0]}, head {xgl - lo:.2f} km) .. {hi:g} km ({rows[hi][0]}, head "
            f"{xgl - hi:.2f} km)")
    log(f"done: the {a.criterion} status switches between d {lo:g} km (head {xgl - lo:.2f} km, {rows[lo][0]}: "
        f"{os.path.relpath(ends[lo], ROOT)}) and d {hi:g} km (head {xgl - hi:.2f} km, {rows[hi][0]}: "
        f"{os.path.relpath(ends[hi], ROOT)}); every point in {os.path.relpath(csvf, ROOT)}")


if __name__ == "__main__":
    main()
