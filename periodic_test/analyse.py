"""
Analysis of the periodic-BC test cases (square slab, periodic on all sides): horizontal uniformity of every
run, agreement with the direct-solver reference of the same physics, values at partition-interface nodes
against interior nodes, disagreement between partitions at shared nodes. Writes summary.csv, and per case
bed.npz (final state) and bed_time.npz (snapshots) for plot_bed.py / plot_bed_time.py.

Nodes are identified by footprint position and layer (x, y, level), not by z, so cases with a free surface
(moving mesh) compare node by node.
"""
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ess.statefile import partition_files, read_state_file  # noqa: E402

KEY_DIGITS = 2       # coordinates rounded to 1 cm to merge the partitions' copies of shared nodes


def load_case(d, base="out.bin"):
    """Nodes merged over the partition files: dict of arrays (x, y, z, level, nparts, owner, variables),
    keys (x, y, level), and per variable the max disagreement between partitions at shared nodes."""
    files = partition_files(os.path.join(d, base))
    if not files:
        return None
    parts = [read_state_file(f) for f in files]
    K = [np.round(np.c_[r["coordinate 1"][1], r["coordinate 2"][1], r["coordinate 3"][1]], KEY_DIGITS)
         for r in parts]
    uniq, inv = np.unique(np.concatenate(K), axis=0, return_inverse=True)     # sorted by x, then y, then z
    inv = inv.ravel()
    n = len(uniq)
    _, col_start, col = np.unique(uniq[:, :2], axis=0, return_index=True, return_inverse=True)
    col = col.ravel()
    level = np.arange(n) - col_start[col]                                     # rank in z within the column
    nparts = np.bincount(inv, minlength=n)
    owner = np.full(n, len(files), int)
    bounds = np.cumsum([0] + [len(k) for k in K])
    node_of = []
    for pe, r in enumerate(parts):
        ip = inv[bounds[pe]:bounds[pe + 1]]
        np.minimum.at(owner, ip, pe)                   # lowest partition holding the node
        g1 = r["coordinate 1"][0]
        srt = np.argsort(g1)
        node_of.append((g1[srt], ip[srt]))
    out = {"x": uniq[:, 0], "y": uniq[:, 1], "z": uniq[:, 2], "level": level, "col": col,
           "nparts": nparts, "owner": owner, "nfiles": len(files),
           "keys": [tuple(k) for k in np.c_[uniq[:, :2], level].tolist()]}
    names = []
    for r in parts:
        names += [k for k in r if not k.startswith("coordinate") and k not in names]
    for name in names:
        val = np.full(n, np.nan)
        seen = np.zeros(n, bool)
        mis = np.zeros(n)
        for (gs, ns), r in zip(node_of, parts):
            if name not in r:
                continue
            g, v = r[name]
            nd = ns[np.searchsorted(gs, g)]
            new = ~seen[nd]
            val[nd[new]] = v[new]
            seen[nd[new]] = True
            old = ~new
            if old.any():
                np.maximum.at(mis, nd[old], np.abs(v[old] - val[nd[old]]))
        out[name] = val
        out[f"{name} mismatch"] = float(np.nanmax(mis)) if n else 0.0
        out[f"{name} node mismatch"] = mis
    return out


def find(D, name):
    k = next((k for k in D if isinstance(k, str) and k.lower() == name.lower()), None)
    return D[k] if k else None


def log_flags(d):
    p = os.path.join(d, "elmer.log")
    if not os.path.exists(p):
        return "no log"
    t = open(p, errors="replace").read()
    flags = []
    if "ALL DONE" not in t:
        flags.append("NOT FINISHED")
    n = len(re.findall(r"Too many iterations|not converged|NOT CONVERGED", t, flags=re.I))
    if n:
        flags.append(f"{n} non-converged linear solves")
    its = re.findall(r"Steady state iteration:\s*(\d+)", t, flags=re.I)
    if its and int(its[-1]) > 1:
        flags.append(f"{its[-1]} coupled iterations")
    err = [l.strip() for l in t.splitlines() if "ERROR" in l]
    if err:
        flags.append(err[-1][:80])
    return "; ".join(flags) or "ok"


def plane_dev(level, v):
    """v minus its mean over the horizontal plane (same level); works on [time, node] arrays"""
    v = np.atleast_2d(np.asarray(v, float))
    d = np.full(v.shape, np.nan)
    for lv in np.unique(level):
        m = level == lv
        d[:, m] = v[:, m] - np.nanmean(v[:, m], axis=1, keepdims=True)
    return d


def plane_spread(level, v):
    """max over levels of (max - min over the horizontal plane), and the node furthest from its plane mean"""
    v = np.asarray(v, float)
    if not np.isfinite(v).any():
        return np.nan, None
    d = plane_dev(level, v)[0]
    sp = 0.0
    for lv in np.unique(level):
        m = (level == lv) & np.isfinite(v)
        if m.sum() >= 2:
            sp = max(sp, float(v[m].max() - v[m].min()))
    wi = int(np.nanargmax(np.abs(d)))
    return sp, wi


def references(CASES, data, physics_key, is_direct):
    """physics -> reference case: serial direct if finished, else the first partitioned direct run"""
    refs = {}
    for c in CASES:
        k = physics_key(c)
        if c["name"] not in data or not is_direct(c):
            continue
        if k not in refs or (c["parts"] == "serial" and refs[k][1] != "serial"):
            refs[k] = (c["name"], c["parts"])
    return {k: v[0] for k, v in refs.items()}


def analyse_all(CASES, physics_key, is_direct, OUT, GEOM):
    L, W = GEOM["L"], GEOM["W"]
    data = {}
    for c in CASES:
        d = os.path.join(OUT, c["name"])
        D = load_case(d) if os.path.isdir(d) else None
        if D is not None:
            data[c["name"]] = D
    refs = references(CASES, data, physics_key, is_direct)
    rows = []
    for c in CASES:
        d = os.path.join(OUT, c["name"])
        if c["name"] not in data:
            if os.path.isdir(d):
                rows.append(dict(case=c["name"], status=log_flags(d) + "; no output"))
            continue
        D = data[c["name"]]
        T, u, v, w = (find(D, n) for n in ("temperature", "velocity 1", "velocity 2", "velocity 3"))
        lev = D["level"]
        iface = D["nparts"] > 1
        side = (np.abs(D["x"]) < 1e-3) | (np.abs(D["x"] - L) < 1e-3) | (np.abs(D["y"]) < 1e-3) | (np.abs(D["y"] - W) < 1e-3)
        bed, top = lev == 0, lev == lev.max()
        sp, wi = plane_spread(lev, T)
        checks = []
        if not c.get("fs") and abs((D["z"].max() - D["z"].min()) / GEOM["H"] - 1) > 0.01:
            checks.append(f"slab {D['z'].max() - D['z'].min():.3g} m thick, not {GEOM['H']:g}")
        if np.nanmax(np.abs(T[top] - c["Ts"])) > 1e-6:
            checks.append(f"surface T not {c['Ts']}: check the extruded BC numbers (top = 6)")
        if w is not None and np.nanmax(np.abs(w[bed])) > 1e-6 * np.nanmax(np.abs(u)):
            checks.append("w at the bed not 0: check the extruded BC numbers (bed = 5)")
        Zs = find(D, "zs")
        row = dict(case=c["name"], parts=c["parts"], nparts=D["nfiles"],
                   status="; ".join([log_flags(d)] + checks),
                   T_spread=sp, Tbed_spread=plane_spread(lev[bed], T[bed])[0],
                   u_spread_rel=plane_spread(lev, u)[0] / (np.nanmax(np.abs(u)) or 1.0),
                   v_max_rel=np.nanmax(np.abs(v)) / (np.nanmax(np.abs(u)) or 1.0) if v is not None else np.nan,
                   Zs_spread=plane_spread(lev[top], Zs[top])[0] if Zs is not None else np.nan,
                   worst_at=(f"x {D['x'][wi] / 1e3:.2f} km y {D['y'][wi] / 1e3:.2f} km level {lev[wi]}"
                             + (" [interface]" if iface[wi] else "") + (" [periodic side]" if side[wi] else ""))
                   if wi is not None else "",
                   T_partition_mismatch=find(D, "temperature mismatch") or 0.0)
        ref = refs.get(physics_key(c))
        R = data[ref] if ref and ref != c["name"] else None
        if R is not None:
            pos = {k: i for i, k in enumerate(R["keys"])}
            j = np.array([pos.get(k, -1) for k in D["keys"]])
            ok = j >= 0
            dT = np.full(len(T), np.nan)
            dT[ok] = T[ok] - find(R, "temperature")[j[ok]]
            du = np.full(len(T), np.nan)
            du[ok] = u[ok] - find(R, "velocity 1")[j[ok]]
            row.update(ref=ref, dT_max=np.nanmax(np.abs(dT)),
                       dT_interface=np.nanmax(np.abs(dT[iface])) if iface.any() else np.nan,
                       dT_interior=np.nanmax(np.abs(dT[~iface & ~side])),
                       dT_periodic_side=np.nanmax(np.abs(dT[side])),
                       du_rel=np.nanmax(np.abs(du)) / (np.nanmax(np.abs(find(R, "velocity 1"))) or 1.0))
        rows.append(row)
        save_bed(d, D, c, row, R)
        if c.get("steps"):
            save_bed_time(d, c, row, os.path.join(OUT, ref) if R is not None else None)
    if not rows:
        print(f"no finished cases under {OUT}")
        return
    cols = ["case", "parts", "nparts", "T_spread", "Tbed_spread", "u_spread_rel", "v_max_rel", "Zs_spread",
            "dT_max", "dT_interface", "dT_interior", "dT_periodic_side", "du_rel", "T_partition_mismatch",
            "worst_at", "ref", "status"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    def fmt(x):
        return f"{x:9.2e}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else f"{'-':>9s}"
    print("\n*_spread: max over levels of the range over the horizontal plane (exact: 0); T in K, Zs in m,")
    print("u relative; v_max: max |v| / max |u| (exact: 0); dT_*: |T - T_ref| (direct solvers, same physics)")
    print("at interface / interior / periodic-side nodes; mismatch: partitions disagreeing at shared nodes\n")
    head = ("T_sprd", "Tbed_sprd", "u_sprd", "v_max", "Zs_sprd", "dT_max", "dT_iface", "dT_inter", "dT_side",
            "du_rel", "mismatch")
    print(f"{'case':24s} " + " ".join(f"{h:>9s}" for h in head) + "  ref / status")
    for r in rows:
        print(f"{r['case']:24s} " + " ".join(fmt(r.get(k)) for k in
              ("T_spread", "Tbed_spread", "u_spread_rel", "v_max_rel", "Zs_spread", "dT_max", "dT_interface",
               "dT_interior", "dT_periodic_side", "du_rel", "T_partition_mismatch"))
              + f"  {r.get('ref', '') or ''}{' / ' if r.get('ref') else ''}{r.get('status', '')}")
    print(f"\nwrote {os.path.join(OUT, 'summary.csv')}")


def footprint_map(D, sel, keys_xy):
    """index of each footprint node (x, y) among the nodes in sel, -1 where missing"""
    pos = {(x, y): i for i, (x, y) in enumerate(zip(np.round(D["x"][sel], KEY_DIGITS), np.round(D["y"][sel], KEY_DIGITS)))}
    return np.array([pos.get(k, -1) for k in keys_xy])


def take(v, j):
    out = np.full(len(j), np.nan)
    out[j >= 0] = v[j[j >= 0]]
    return out


def meta(c, row):
    return dict(case=c["name"], status=str(row.get("status", "")), parts=c["parts"], heat=c["heat"],
                halve=c["halve"], ls_heat=c["ls_heat"], ls_stokes=c["ls_stokes"], ls_fs=c["ls_fs"],
                element=c["element"], fs=c["fs"], fssa=c["fssa"], dt=c["dt"])


def save_bed(d, D, c, row, R):
    """bed.npz: final basal fields (and surface elevation) on the footprint, for plot_bed.py"""
    bed, top = D["level"] == 0, D["level"] == D["level"].max()
    x, y = D["x"][bed], D["y"][bed]
    T = find(D, "temperature")
    out = dict(x=x, y=y, T=T[bed], nparts=D["nparts"][bed], owner=D["owner"][bed], nfiles=D["nfiles"], **meta(c, row))
    mis = find(D, "temperature node mismatch")
    if mis is not None:
        out["T_mismatch"] = mis[bed]
    u = [find(D, f"velocity {i}") for i in (1, 2, 3)]
    l = [find(D, f"flow solution loads {i}") for i in (1, 2, 3)]
    if all(v is not None for v in u):
        out["ub"] = np.sqrt(sum(v[bed] ** 2 for v in u))
    if all(v is not None for v in u + l):
        out["fric_load"] = sum(np.maximum(-li[bed] * ui[bed], 0) for li, ui in zip(l, u))
    Zs = find(D, "zs")
    if Zs is not None:
        out["Zs"] = take(Zs[top], footprint_map(D, top, list(zip(np.round(x, KEY_DIGITS), np.round(y, KEY_DIGITS)))))
    if R is not None:
        pos = {k: i for i, k in enumerate(R["keys"])}
        j = np.array([pos.get(k, -1) for k, b in zip(D["keys"], bed) if b])
        out["dT_ref"] = T[bed] - take(find(R, "temperature"), j)
        out["ref"] = row.get("ref", "")
    np.savez(os.path.join(d, "bed.npz"), **out)


def snapshot_steps(d):
    steps = set()
    for f in glob.glob(os.path.join(d, "snap_t*")):
        m = re.match(r"snap_t(\d+)(?:\.bin)?\.\d+$", os.path.basename(f))   # snap_t<step>.<pe>
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


_SERIES = {}


def bed_series(d):
    """per snapshot: basal T and surface Zs on the footprint nodes of the first snapshot"""
    if d in _SERIES:
        return _SERIES[d]
    steps = snapshot_steps(d)
    xy, Ts, Zs, info, used = None, [], [], None, []
    for k in steps:
        D = load_case(d, f"snap_t{k}") or load_case(d, f"snap_t{k}.bin")
        if D is None:
            continue
        bed, top = D["level"] == 0, D["level"] == D["level"].max()
        if xy is None:
            xy = list(zip(np.round(D["x"][bed], KEY_DIGITS), np.round(D["y"][bed], KEY_DIGITS)))
            info = dict(x=D["x"][bed], y=D["y"][bed], nparts=D["nparts"][bed], owner=D["owner"][bed],
                        nfiles=D["nfiles"])
        Ts.append(take(find(D, "temperature")[bed], footprint_map(D, bed, xy)))
        z = find(D, "zs")
        if z is not None:
            Zs.append(take(z[top], footprint_map(D, top, xy)))
        used.append(k)
    r = (xy, used, np.array(Ts) if Ts else None, np.array(Zs) if Zs and len(Zs) == len(Ts) else None, info)
    _SERIES[d] = r
    return r


def save_bed_time(d, c, row, refdir):
    """bed_time.npz: basal T (and Zs) at every snapshot, and T minus the reference's, for plot_bed_time.py"""
    xy, steps, T, Zs, info = bed_series(d)
    if T is None:
        return
    out = dict(steps=np.array(steps), time=np.array(steps) * c["dt"], T=T, **info, **meta(c, row))
    if Zs is not None:
        out["Zs"] = Zs
    if refdir:
        rxy, rsteps, RT, _, _ = bed_series(refdir)
        if RT is not None:
            rp = {k: i for i, k in enumerate(rxy)}
            j = np.array([rp.get(k, -1) for k in xy])
            dT = np.full(T.shape, np.nan)
            for a, k in enumerate(steps):
                if k in rsteps:
                    dT[a] = T[a] - take(RT[rsteps.index(k)], j)
            out["dT_ref"] = dT
            out["ref"] = os.path.basename(refdir)
    np.savez(os.path.join(d, "bed_time.npz"), **out)
