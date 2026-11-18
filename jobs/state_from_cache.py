#!/usr/bin/env python3
"""A driver state (.npz) from the newest finished chunk of a running 2D bound (jobs/flowline_bounds.cmd) in the Elmer
runs' cache, e.g. to start Newton from it before the job ends: the state vector of the cache file, the layout and the
frozen bed and cold/temperate mask of REF, the grounded mask from the chunk's own diagnostics (the contact grounding
line moved it). Which bound a cache file is: as plots/bounds_progress.py (median basal speed on the grounded bed below
--cold-speed, 1 m/yr: cold).

    python3 jobs/state_from_cache.py WORKDIR REF.npz OUT.npz --which temperate|cold [--since "2026-11-16 20:20"]
    python3 jobs/state_from_cache.py WORKDIR REF.npz OUT.npz --at "2026-11-18 00:14"

--at: instead, the cache file written closest to that minute (any run, e.g. a Newton solve's last evaluation: its
output is the converged state to within the final rms(F)).
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import find_name, load_state, save_state  # noqa: E402


def arg(args, name, default):
    if name in args:
        k = args.index(name)
        v = args[k + 1]
        del args[k:k + 2]
        return v
    return default


def main():
    args = sys.argv[1:]
    which = arg(args, "--which", None)
    at = arg(args, "--at", None)
    since = arg(args, "--since", "2026-11-16 20:20")
    cold_speed = float(arg(args, "--cold-speed", "1.0"))
    if len(args) != 3 or (at is None and which not in ("temperate", "cold")):
        raise SystemExit(__doc__)
    work, ref, out = args
    t_since = time.mktime(time.strptime(since, "%Y-%m-%d %H:%M"))
    s = load_state(ref)
    zb = next(b for b in s.layout.blocks if b.name.lower() == "zb")
    files = sorted((f for f in glob.glob(os.path.join(work, "cache", "*.npz"))
                    if not f.endswith(".tmp.npz") and os.path.getmtime(f) > t_since), key=os.path.getmtime)
    pick = None
    if at is not None:
        sec = at.count(":") == 2                     # with seconds: that instant; without: the middle of the minute
        t_at = time.mktime(time.strptime(at, "%Y-%m-%d %H:%M:%S" if sec else "%Y-%m-%d %H:%M")) + (0 if sec else 30)
        cand = sorted((f for f in glob.glob(os.path.join(work, "cache", "*.npz")) if not f.endswith(".tmp.npz")),
                      key=lambda f: abs(os.path.getmtime(f) - t_at))
        print("cache files nearest: " + ", ".join(
            f"{os.path.basename(c)[:10]} {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(c)))}"
            for c in cand[:3]))
        f = cand[0]
        with np.load(f) as z:
            xc = z["x"].copy()
            dg = {k[3:]: (z["g::" + k[3:]], z[k]) for k in z.files if k.startswith("v::")}
        if len(xc) != s.layout.size:
            # a mechanics-only run (ess newton/run --stokes-only): its state is REF's without Temperature (moved to the
            # frozen fields): the geometry from the cache, the temperature REF's
            mech = [b for b in s.layout.blocks if b.name.lower() != "temperature"]
            if len(xc) != sum(len(b.gids) for b in mech):
                raise SystemExit(f"{f}: neither REF's layout ({s.layout.size}) nor its geometry blocks "
                                 f"({sum(len(b.gids) for b in mech)}): {len(xc)} values")
            parts, i = {}, 0
            for b in mech:
                parts[b.name] = xc[i:i + len(b.gids)]
                i += len(b.gids)
            full = s.layout.split(s.x)
            xc = np.concatenate([parts.get(b.name, full[b.name]) for b in s.layout.blocks])
            print(f"{os.path.basename(f)}: a mechanics-only state (Zs, Zb); the temperature from REF")
        pick = (f, xc, dg)
        which = "at " + at
        files = []
    for f in reversed(files):
        with np.load(f) as z:
            if len(z["x"]) != s.layout.size:
                continue
            diag = {k[3:]: (z["g::" + k[3:]], z[k]) for k in z.files if k.startswith("v::")}
            gu, vu = diag[find_name(diag, "velocity 1")]
            U = dict(zip(gu.tolist(), vu))
            gg, vg = diag[find_name(diag, "groundedmask")]
            G = dict(zip(gg.tolist(), vg))
            ub = np.array([U.get(g, np.nan) for g in zb.gids])
            gm = np.array([G.get(g, np.nan) for g in zb.gids])
            grounded = gm > -0.5
            if not grounded.any():
                continue
            w = "cold" if np.nanmedian(np.abs(ub[grounded])) < cold_speed else "temperate"
            if w == which:
                pick = (f, z["x"].copy(), diag)
                break
    if pick is None:
        raise SystemExit(f"no {which} chunk in {work}/cache since {since}")
    f, x, diag = pick
    frozen = dict(s.frozen)
    kg = find_name(frozen, "groundedmask")
    gg, vg = diag[find_name(diag, "groundedmask")]
    G = dict(zip(gg.tolist(), vg))
    g0, v0 = frozen[kg]
    frozen[kg] = (g0, np.array([G.get(g, v) for g, v in zip(g0.tolist(), v0)]))
    save_state(out, s.layout, x, s.p, frozen, diag, source=os.path.basename(f))
    print(f"{which}: {os.path.basename(f)} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))}) "
          f"-> {out}")


if __name__ == "__main__":
    main()
