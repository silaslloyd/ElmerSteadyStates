#!/usr/bin/env python3
"""
Compare the Stokes solver tests (jobs/stokes_tests.sh): for each geometry folder (work/test_stokes_land,
work/test_stokes_marine), the runs run_direct.npz, run_iterative.npz, run_block.npz, run_amg.npz (same mesh, same start, same
steps; only the Stokes linear solver differs).

Prints, per run: Elmer seconds per time step, and the differences from the direct (MUMPS) run: surface Zs, and the
along-flow velocity u (all nodes; marine also grounded / floating separately, by the grounded mask at the bed
below each node), max and rms, absolute and relative to max |u| of the direct run.
Writes <folder>/stokes_tests.html: the surface velocity u(x) and Zs(x) of the three runs (plotly, if installed).

    python plots/compare_stokes_tests.py                         (both folders)
    python plots/compare_stokes_tests.py work/test_stokes_marine
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from ess.masks import _on  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402

SOLVERS = ("direct", "iterative", "block", "amg")
SHOW = True


def fields(s):
    d = s.diag
    g1, u = d[find_name(d, "velocity 1")]
    gx, X = d[find_name(d, "coordinate 1")]
    gy, Y = d[find_name(d, "coordinate 2")]
    x = _on(gx, X, g1)
    y = _on(gy, Y, g1)
    zb = next(b for b in s.layout.blocks if b.name.lower() == "zs")
    zs = s.layout.split(s.x)[zb.name]
    xz = _on(gx, X, zb.gids)
    floating = np.zeros(len(g1), bool)
    if any(k.lower() == "groundedmask" for k in s.frozen):
        gg, gm = s.frozen[find_name(s.frozen, "GroundedMask")]
        xg = _on(gx, X, gg)
        if (gm < -0.5).any() and (gm > -0.5).any():
            x_gl = xg[gm > -0.5].max()
            floating = x > x_gl + 1.0
    return dict(g=g1, u=u, x=x, y=y, zs=zs, gz=zb.gids, xz=xz, floating=floating)


def stats(a, b, sel=None):
    d = a - b if sel is None else (a - b)[sel]
    if d.size == 0:
        return np.nan, np.nan
    return float(np.max(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


def compare(folder):
    runs = {}
    names = sorted(os.path.basename(f)[4:-4] for f in glob.glob(os.path.join(folder, "run_*.npz")))
    order = {sv: i for i, sv in enumerate(SOLVERS)}
    for sv in sorted(names, key=lambda n: (n.endswith("_cart"), order.get(n.replace("_cart", ""), 9), n)):
        runs[sv] = load_state(os.path.join(folder, f"run_{sv}.npz"))
    print(f"\n{folder}: " + (", ".join(runs) if runs else "no runs yet"))
    if not runs:
        return
    F = {sv: fields(s) for sv, s in runs.items()}
    for sv, s in runs.items():
        m = s.meta
        sec, st = float(m.get("elmer_seconds", np.nan)), int(m.get("steps", 0))
        print(f"  {sv:9s}: {st} steps, Elmer {sec:.0f} s" + (f", {sec / st:.2f} s per step" if st else "")
              + f", t = {float(m.get('time', np.nan)):g} yr")
    for sv in F:                         # each run against the direct one with the same bed condition
        ref = "direct_cart" if sv.endswith("_cart") else "direct"
        if sv == ref:
            continue
        if ref not in F:
            print(f"  {sv}: no {ref} run to compare with")
            continue
        R, A = F[ref], F[sv]
        umax = float(np.max(np.abs(R["u"]))) or 1.0
        marine = R["floating"].any()
        if not np.array_equal(A["g"], R["g"]):
            print(f"  {sv}: different nodes from the direct run (not the same mesh?)")
            continue
        zmax, zrms = stats(A["zs"], R["zs"])
        umx, urms = stats(A["u"], R["u"])
        line = (f"  {sv:9s} - {ref}: Zs max {zmax:.3g} m, rms {zrms:.3g} m; u max {umx:.3g} m/yr "
                f"({umx / umax:.2%} of max |u|), rms {urms:.3g} m/yr")
        if marine:
            fl = R["floating"]
            gmx, _ = stats(A["u"], R["u"], ~fl)
            fmx, frms = stats(A["u"], R["u"], fl)
            ufl = float(np.max(np.abs(R["u"][fl])))
            line += (f"\n              grounded: u max {gmx:.3g} m/yr; floating: u max {fmx:.3g} m/yr, rms {frms:.3g} "
                     f"({fmx / ufl:.2%} of max floating |u| {ufl:.4g}); mean floating u "
                     f"{np.mean(A['u'][fl]):.4g} vs {ref} {np.mean(R['u'][fl]):.4g} m/yr")
        print(line)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  (plotly not installed: no html)")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["along-flow velocity at the surface (m/yr)", "surface elevation Zs (m)"])
    cols = {"direct": "black", "iterative": "#e67e22", "block": "#2980b9", "amg": "#27ae60"}
    cols.update({k + "_cart": v for k, v in list(cols.items())})
    for sv, A in F.items():
        # surface u: the node with the largest z per column is not stored; use the Zs nodes' u
        us = _on(A["g"], A["u"], A["gz"])
        o = np.argsort(A["xz"])
        dash = None if sv.startswith("direct") else "dash"
        fig.add_trace(go.Scatter(x=A["xz"][o] / 1e3, y=us[o], mode="lines", name=f"u {sv}",
                                 line=dict(color=cols[sv], dash=dash)), 1, 1)
        fig.add_trace(go.Scatter(x=A["xz"][o] / 1e3, y=A["zs"][o], mode="lines", name=f"Zs {sv}",
                                 line=dict(color=cols[sv], dash=dash)), 2, 1)
    fig.update_xaxes(title_text="x (km)", row=2, col=1)
    fig.update_layout(title=os.path.basename(folder), height=800, template="plotly_white")
    out = os.path.join(folder, "stokes_tests.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  wrote {out}")
    if SHOW:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out))


def main():
    folders = sys.argv[1:] or [os.path.join(REPO, "work", "test_stokes_land"), os.path.join(REPO, "work", "test_stokes_marine")]
    for f in folders:
        compare(f if os.path.isabs(f) else os.path.join(REPO, f) if not os.path.exists(f) else f)


if __name__ == "__main__":
    main()
