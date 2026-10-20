"""
Analysis of the periodic-BC test cases: y-uniformity of every run, agreement with the serial direct
reference of the same physics, values at partition-interface nodes against interior nodes, and
disagreement between partitions at shared nodes. Writes out/summary.csv and, if plotly is installed,
out/<case>/bed.html (basal temperature minus its across-flow mean, partition interfaces marked).
"""
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ess.statefile import partition_files, read_state_file  # noqa: E402

KEY_DIGITS = 2       # coordinates rounded to 1 cm to identify nodes across runs and partitions


def load_case(d, base="out.bin"):
    """Nodes keyed by rounded coordinates: dict of arrays + 'nparts' (how many partition files hold
    each node) + max disagreement between partitions at shared nodes, per variable."""
    files = partition_files(os.path.join(d, base))
    if not files:
        return None
    acc, cnt, spread = {}, {}, {}
    xyz, owner = {}, {}
    for pe, f in enumerate(files):
        r = read_state_file(f)
        g1, x = r["coordinate 1"]
        _, y = r["coordinate 2"]
        _, z = r["coordinate 3"]
        keys = list(zip(np.round(x, KEY_DIGITS), np.round(y, KEY_DIGITS), np.round(z, KEY_DIGITS)))
        pos = {g: k for g, k in zip(g1, keys)}
        for k in keys:
            cnt[k] = cnt.get(k, 0) + 1
            xyz[k] = k
            owner.setdefault(k, pe)                # lowest partition holding the node
        for name, (g, v) in r.items():
            if name.startswith("coordinate"):
                continue
            a = acc.setdefault(name, {})
            s = spread.setdefault(name, {})
            for gi, vi in zip(g, v):
                k = pos[gi]
                if k in a:
                    s[k] = max(s.get(k, 0.0), abs(vi - a[k]))
                else:
                    a[k] = vi
    keys = sorted(xyz)
    out = {"x": np.array([k[0] for k in keys]), "y": np.array([k[1] for k in keys]),
           "z": np.array([k[2] for k in keys]), "nparts": np.array([cnt[k] for k in keys]),
           "owner": np.array([owner[k] for k in keys]),
           "keys": keys, "nfiles": len(files)}
    for name, a in acc.items():
        out[name] = np.array([a.get(k, np.nan) for k in keys])
        out[f"{name} mismatch"] = max(spread[name].values()) if spread[name] else 0.0
        out[f"{name} node mismatch"] = np.array([spread[name].get(k, 0.0) for k in keys])
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


def column_spread(D, v):
    """max over (x, z) of (max - min over y) of v, and the node of the largest deviation from its
    column mean"""
    cols = {}
    for i, (x, z) in enumerate(zip(D["x"], D["z"])):
        cols.setdefault((x, round(z, 0)), []).append(i)
    best, worst, wi = 0.0, 0.0, None
    for idx in cols.values():
        vv = v[idx]
        if len(vv) < 2 or np.all(np.isnan(vv)):
            continue
        best = max(best, np.nanmax(vv) - np.nanmin(vv))
        dev = np.abs(vv - np.nanmean(vv))
        j = int(np.nanargmax(dev))
        if dev[j] > worst:
            worst, wi = dev[j], idx[j]
    return best, wi


def analyse_all(CASES, PARTS, physics_key, OUT, GEOM):
    W = GEOM["W"]
    data = {}
    for c in CASES:
        d = os.path.join(OUT, c["name"])
        D = load_case(d) if os.path.isdir(d) else None
        if D is not None:
            data[c["name"]] = D
    is_ref = lambda c: c["parts"] == "serial" and c["ls_heat"] == "direct" and c["ls_stokes"] == "direct"
    refs = {physics_key(c): c["name"] for c in CASES if is_ref(c) and c["name"] in data}
    rows = []
    for c in CASES:
        if c["name"] not in data:
            if os.path.isdir(os.path.join(OUT, c["name"])):
                rows.append(dict(case=c["name"], status=log_flags(os.path.join(OUT, c["name"])) + "; no output"))
            continue
        D = data[c["name"]]
        T, u = find(D, "temperature"), find(D, "velocity 1")
        iface = D["nparts"] > 1
        side = (np.abs(D["y"]) < 1e-3) | (np.abs(D["y"] - W) < 1e-3)
        bed = np.abs(D["z"] - D["z"].min()) < 1e-3
        sp, wi = column_spread(D, T)
        top = np.abs(D["z"] - D["z"].max()) < 1e-3
        checks = []                                       # did the extruded bed/surface BCs (5, 6) apply?
        if abs((D["z"].max() - D["z"].min()) / GEOM["H"] - 1) > 0.01:
            checks.append(f"slab {D['z'].max() - D['z'].min():.3g} m thick, not {GEOM['H']:g} (Extruded Max Coordinate?)")
        Ts = c.get("Ts", 243.15)
        if np.nanmax(np.abs(T[top] - Ts)) > 1e-6:
            checks.append(f"surface T not {Ts}: check the extruded BC numbers (top = 6)")
        w = find(D, "velocity 3")
        if w is not None and np.nanmax(np.abs(w[bed])) > 1e-6 * np.nanmax(np.abs(u)):
            checks.append("w at the bed not 0: check the extruded BC numbers (bed = 5)")
        row = dict(case=c["name"], parts=c["parts"], nparts=D["nfiles"],
                   status="; ".join([log_flags(os.path.join(OUT, c["name"]))] + checks),
                   T_y_spread=sp, Tbed_y_spread=column_spread({**D, "x": D["x"][bed], "z": D["z"][bed]}, T[bed])[0],
                   u_y_spread=column_spread(D, u)[0],
                   sh_y_spread_rel=(column_spread(D, find(D, "strain heating"))[0]
                                    / (np.nanmax(np.abs(find(D, "strain heating"))) or 1.0))
                   if find(D, "strain heating") is not None and np.nanmax(np.abs(find(D, "strain heating"))) > 0
                   else np.nan,
                   worst_at=(f"x {D['x'][wi] / 1e3:.2f} km y {D['y'][wi] / 1e3:.2f} km z {D['z'][wi]:.0f} m"
                             + (" [interface]" if iface[wi] else "") + (" [periodic side]" if side[wi] else ""))
                   if wi is not None else "",
                   T_partition_mismatch=find(D, "temperature mismatch") or 0.0)
        ref = refs.get(physics_key(c))
        if ref and ref != c["name"]:
            R = data[ref]
            pos = {k: i for i, k in enumerate(R["keys"])}
            j = np.array([pos.get(k, -1) for k in D["keys"]])
            ok = j >= 0
            dT = np.full(len(T), np.nan)
            dT[ok] = T[ok] - find(R, "temperature")[j[ok]]
            du = np.full(len(T), np.nan)
            du[ok] = u[ok] - find(R, "velocity 1")[j[ok]]
            umax = np.nanmax(np.abs(find(R, "velocity 1")))
            row.update(ref=ref, dT_max=np.nanmax(np.abs(dT)),
                       dT_interface=np.nanmax(np.abs(dT[iface])) if iface.any() else np.nan,
                       dT_interior=np.nanmax(np.abs(dT[~iface & ~side])),
                       dT_periodic_side=np.nanmax(np.abs(dT[side])),
                       du_rel=np.nanmax(np.abs(du)) / umax)
        rows.append(row)
        plot_bed(os.path.join(OUT, c["name"]), D, T, bed, iface, c["name"], row)
        save_bed(os.path.join(OUT, c["name"]), D, T, bed, c, row, data.get(ref) if ref and ref != c["name"] else None)
        if c.get("steps"):
            save_bed_time(os.path.join(OUT, c["name"]), c, row,
                          os.path.join(OUT, ref) if ref and ref != c["name"] else None)
    cols = ["case", "parts", "nparts", "T_y_spread", "Tbed_y_spread", "u_y_spread", "sh_y_spread_rel", "dT_max", "dT_interface",
            "dT_interior", "dT_periodic_side", "du_rel", "T_partition_mismatch", "worst_at", "ref", "status"]
    if not rows:
        print(f"no finished cases under {OUT}")
        return
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return f"{v:9.2e}" if isinstance(v, float) and np.isfinite(v) else f"{'-':>9s}"
    print("\nK and relative velocity; *_y_spread: max over (x, z) of the range over y (exact: 0);")
    print("dT_*: |T - T_ref| (serial, direct, same physics) at interface / interior / periodic-side nodes\n")
    print(f"{'case':38s} {'T_y_sprd':>9s} {'Tbed_y':>9s} {'u_y_sprd':>9s} {'dT_max':>9s} {'dT_iface':>9s} "
          f"{'dT_inter':>9s} {'dT_side':>9s} {'du_rel':>9s} {'mismatch':>9s}  status")
    for r in rows:
        print(f"{r['case']:38s} " + " ".join(fmt(r.get(k)) for k in
              ("T_y_spread", "Tbed_y_spread", "u_y_spread", "dT_max", "dT_interface", "dT_interior",
               "dT_periodic_side", "du_rel", "T_partition_mismatch")) + f"  {r.get('status', '')}")
    print(f"\nwrote {os.path.join(OUT, 'summary.csv')}")


def save_bed(d, D, T, bed, c, row, R):
    """bed.npz: basal fields on the footprint nodes, for plot_bed.py (copy it to a laptop and look)."""
    x, y = D["x"][bed], D["y"][bed]
    out = dict(x=x, y=y, T=T[bed], nparts=D["nparts"][bed], owner=D["owner"][bed], nfiles=D["nfiles"],
               case=c["name"], status=str(row.get("status", "")), parts=c["parts"], bc=c["bc"], heat=c["heat"],
               halve=c["halve"], ls_heat=c["ls_heat"], ls_stokes=c["ls_stokes"], element=c["element"])
    mis = find(D, "temperature node mismatch")
    if mis is not None:
        out["T_mismatch"] = mis[bed]
    u = [find(D, f"velocity {i}") for i in (1, 2, 3)]
    l = [find(D, f"flow solution loads {i}") for i in (1, 2, 3)]
    if all(v is not None for v in u):
        out["ub"] = np.sqrt(sum(v[bed] ** 2 for v in u))
    if all(v is not None for v in u + l):         # the nodal friction load (frictionloads, no halving)
        out["fric_load"] = sum(np.maximum(-li[bed] * ui[bed], 0) for li, ui in zip(l, u))
    if R is not None:
        pos = {k: i for i, k in enumerate(R["keys"])}
        kb = [k for k, b in zip(D["keys"], bed) if b]
        j = np.array([pos.get(k, -1) for k in kb])
        Tr = find(R, "temperature")
        dT = np.full(len(kb), np.nan)
        dT[j >= 0] = T[bed][j >= 0] - Tr[j[j >= 0]]
        out["dT_ref"] = dT
        out["ref"] = row.get("ref", "")
    np.savez(os.path.join(d, "bed.npz"), **out)


def snapshot_steps(d):
    steps = set()
    for f in glob.glob(os.path.join(d, "snap_t*")):
        m = re.match(r"snap_t(\d+)(?:\.bin)?\.\d+$", os.path.basename(f))   # snap_t<step>.<pe>
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


def bed_series(d):
    """basal temperature at every snapshot: keys of the bed nodes, steps, T[step, node], node info"""
    steps = snapshot_steps(d)
    keys, Ts, info = None, [], None
    for k in steps:
        D = load_case(d, f"snap_t{k}") or load_case(d, f"snap_t{k}.bin")
        if D is None:
            continue
        bed = np.abs(D["z"] - D["z"].min()) < 1e-3
        kb = [key for key, b in zip(D["keys"], bed) if b]
        if keys is None:
            keys = kb
            info = dict(x=D["x"][bed], y=D["y"][bed], nparts=D["nparts"][bed], owner=D["owner"][bed],
                        nfiles=D["nfiles"])
        pos = {key: i for i, key in enumerate(kb)}
        T = find(D, "temperature")[bed]
        Ts.append(np.array([T[pos[key]] if key in pos else np.nan for key in keys]))
    return keys, [k for k in steps], (np.array(Ts) if Ts else None), info


def save_bed_time(d, c, row, refdir):
    """bed_time.npz: basal temperature at every snapshot (and minus the reference's), for plot_bed_time.py"""
    keys, steps, T, info = bed_series(d)
    if T is None:
        return
    out = dict(steps=np.array(steps), time=np.array(steps) * c["dt"], T=T, **info, case=c["name"],
               status=str(row.get("status", "")), parts=c["parts"], heat=c["heat"], halve=c["halve"],
               ls_heat=c["ls_heat"], ls_stokes=c["ls_stokes"])
    if refdir:
        rkeys, rsteps, RT, _ = bed_series(refdir)
        if RT is not None:
            rp = {key: i for i, key in enumerate(rkeys)}
            j = np.array([rp.get(key, -1) for key in keys])
            dT = np.full(T.shape, np.nan)
            for a, k in enumerate(steps):
                if k in rsteps:
                    b = rsteps.index(k)
                    dT[a, j >= 0] = T[a, j >= 0] - RT[b, j[j >= 0]]
            out["dT_ref"] = dT
            out["ref"] = os.path.basename(refdir)
    np.savez(os.path.join(d, "bed_time.npz"), **out)


def plot_bed(d, D, T, bed, iface, name, row):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    x, y, t, f = D["x"][bed] / 1e3, D["y"][bed] / 1e3, T[bed], iface[bed]
    dev = t.copy()
    for xv in np.unique(x):
        m = x == xv
        dev[m] = t[m] - np.nanmean(t[m])
    lim = np.nanmax(np.abs(dev)) or 1e-12
    xm = np.unique(x)[len(np.unique(x)) // 2]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, subplot_titles=(
        "basal T minus its across-flow mean (K); black rings: partition-interface nodes",
        f"basal T across the flow at x = {xm:.1f} km"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=dev, colorscale="RdBu_r", cmin=-lim, cmax=lim,
                                                                  size=7, colorbar=dict(title="K", len=0.45, y=0.78)),
                             hovertemplate="x %{x:.2f} km<br>y %{y:.2f} km<br>%{marker.color:.3e} K<extra></extra>"), 1, 1)
    if f.any():
        fig.add_trace(go.Scatter(x=x[f], y=y[f], mode="markers", hoverinfo="skip",
                                 marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color="black", width=1))), 1, 1)
    m = x == xm
    o = np.argsort(y[m])
    fig.add_trace(go.Scatter(x=y[m][o], y=t[m][o], mode="lines+markers"), 2, 1)
    fig.update_xaxes(title_text="x (km)", row=1, col=1)
    fig.update_yaxes(title_text="y (km)", row=1, col=1)
    fig.update_xaxes(title_text="y (km)", row=2, col=1)
    fig.update_yaxes(title_text="T bed (K)", row=2, col=1)
    sp = row.get("Tbed_y_spread", np.nan)
    fig.update_layout(title=f"{name}: basal T y-spread {sp:.2e} K, {row.get('status', '')}", height=850,
                      template="plotly_white", showlegend=False)
    fig.write_html(os.path.join(d, "bed.html"), include_plotlyjs="cdn")
