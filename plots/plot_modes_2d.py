#!/usr/bin/env python3
"""The eigenmodes of a flowline `ess stability` result (stability.npz) as pictures: for each mode, its temperature part
over the section (ice, and with --rock the bedrock), and its Zs and Zb parts along x; a slider picks the mode. The
title gives |mu|, the e-folding time and, for complex modes, the period. Eigenvectors have arbitrary scale and sign:
each mode is shown normalised (its largest temperature component 1, Zs/Zb by their own largest); a complex mode as two
slider entries, its real part and its imaginary part (the mode a quarter period later; --imag: the imaginary first).

    python3 plots/plot_modes_2d.py work/<case>/stability/<name>/stability.npz [--modes 8] [--rock] [--imag]
                                   [--nz 80] [--nx 600] [--x XMIN XMAX] [--out FILE.html]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from plot_section import binned, opt  # noqa: E402
from ess.statefile import StateLayout  # noqa: E402


def main():
    args = sys.argv[1:]
    args, nm = opt(args, "--modes", conv=int, default=8)
    args, nz = opt(args, "--nz", conv=int, default=80)
    args, nx = opt(args, "--nx", conv=int, default=600)
    args, xwin = opt(args, "--x", n=2)
    args, out = opt(args, "--out", conv=str)
    rock = "--rock" in args
    imag = "--imag" in args
    args = [a for a in args if a not in ("--rock", "--imag")]
    if len(args) != 1:
        raise SystemExit(__doc__)
    z = np.load(args[0])
    lay = StateLayout.from_npz(z)
    mu, sigma, V, conv = z["mu"], z["sigma"], z["vectors"], z["converged"]
    rec = lambda pre: {k[len(pre):]: (z[k], z[k.replace("gids", "vals")]) for k in z.files if k.startswith(pre)}
    diag, frozen = rec("diag_gids::"), rec("frozen_gids::")
    look = lambda d, name: next(v for k, v in d.items() if k.lower() == name)
    gx, X = look(diag, "coordinate 1")
    gz, Z = look(diag, "coordinate 3")
    gb, B = look(frozen, "bedrock")
    xo, zo, bo = (dict(zip(g.tolist(), v)) for g, v in ((gx, X), (gz, Z), (gb, B)))
    sl = lay.slices()
    blk = {b.name.lower(): b for b in lay.blocks}
    bT = blk["temperature"]
    xT = np.array([xo.get(g, np.nan) for g in bT.gids]) / 1e3
    zT = np.array([zo.get(g, np.nan) for g in bT.gids])
    inrock = np.array([zo.get(g, np.nan) < bo.get(g, np.nan) - 0.5 for g in bT.gids])
    keepT = np.isfinite(xT) & np.isfinite(zT) & (rock | ~inrock)
    if xwin:
        keepT &= (xT >= xwin[0]) & (xT <= xwin[1])

    def prof(name, v):
        b = blk.get(name)
        if b is None:
            return None
        xs = np.array([xo.get(g, np.nan) for g in b.gids]) / 1e3
        vv = v[sl[b.name]] / b.scale
        k = np.isfinite(xs)
        o = np.argsort(xs[k])
        xs, vv = xs[k][o], vv[k][o]
        xu, iu = np.unique(np.round(xs, 3), return_index=True)
        return xu, vv[iu]

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    n = min(nm, V.shape[1])
    T = float(z["dt"]) * int(z["n_steps"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06,
                        subplot_titles=("temperature part (normalised)", "Zs and Zb parts (normalised)"))
    titles, per, labels = [], [], []
    entries = []                                 # (mode, part): a complex mode both its real and imaginary parts
    for i in range(n):
        cplx = np.any(V[:, i].imag) and np.any(V[:, i].real)
        entries += [(i, "im"), (i, "re")] if (cplx and imag) else [(i, "re"), (i, "im")] if cplx else [(i, "")]
    for e, (i, part) in enumerate(entries):
        v = V[:, i].imag if part == "im" else V[:, i].real
        if not np.any(v):
            v = V[:, i].imag
        vT = v[sl[bT.name]] / bT.scale
        a = np.nanmax(np.abs(vT[keepT])) or 1.0
        xg, zg, G = binned(xT[keepT], zT[keepT], vT[keepT] / a, nz, nx)
        tr = [go.Heatmap(x=xg, y=zg, z=G, colorscale="RdBu_r", zmin=-1, zmax=1, zsmooth=False, visible=(e == 0),
                         colorbar=dict(title="T part", len=0.65, y=0.67), hoverongaps=False,
                         hovertemplate="x %{x:.1f} km<br>z %{y:.0f} m<br>%{z:.3f}<extra></extra>")]
        lines = []
        for name, col in (("zs", "black"), ("zb", "#8e44ad")):
            p = prof(name, v)
            if p is None:
                continue
            xs, vv = p
            if xwin:
                m = (xs >= xwin[0]) & (xs <= xwin[1])
                xs, vv = xs[m], vv[m]
            s = np.nanmax(np.abs(vv)) or 1.0
            lines.append(go.Scatter(x=xs, y=vv / s, mode="lines", name=name.capitalize(), visible=(e == 0),
                                    line=dict(color=col, width=1.5), showlegend=(e == 0)))
        per.append((tr, lines))
        sg = sigma[i]
        ef = (f"grows x e in {1 / sg.real:,.0f} yr" if sg.real > 0 else f"decays x 1/e in {-1 / sg.real:,.0f} yr")
        osc = f", period {2 * np.pi / abs(sg.imag):,.0f} yr" if abs(sg.imag) > 1e-12 else ""
        titles.append(f"mode {i}: |mu| {abs(mu[i]):.6f} over {T:g} yr, {ef}{osc}"
                      + ("" if conv[i] else " (NOT converged)")
                      + {"re": " [real part]", "im": " [imaginary part: a quarter period on]", "": ""}[part])
        labels.append(f"{i}{' ' + part if part else ''}")
    idx = []
    for tr, lines in per:
        k0 = len(fig.data)
        for t in tr:
            fig.add_trace(t, row=1, col=1)
        for t in lines:
            fig.add_trace(t, row=2, col=1)
        idx.append(list(range(k0, len(fig.data))))
    steps = []
    for i in range(len(entries)):
        vis = [False] * len(fig.data)
        for k in idx[i]:
            vis[k] = True
        steps.append(dict(method="update", label=labels[i], args=[{"visible": vis}, {"title": titles[i]}]))
    fig.update_layout(title=titles[0], template="plotly_white", height=850,
                      sliders=[dict(active=0, currentvalue=dict(prefix="mode "), steps=steps, pad=dict(t=40))])
    fig.update_yaxes(title_text="z (m)", row=1, col=1)
    fig.update_yaxes(title_text="part", row=2, col=1)
    fig.update_xaxes(title_text="x (km)", row=2, col=1)
    out = out or os.path.join(os.path.dirname(args[0]), "modes.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
