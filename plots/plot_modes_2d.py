#!/usr/bin/env python3
"""The eigenmodes of a flowline `ess stability` result (stability.npz) as pictures: for each mode, its temperature part
over the section (ice, and with --rock the bedrock), and its Zs and Zb parts along x; a slider picks the mode. The
title gives |mu|, the e-folding time and, for complex modes, the period. Eigenvectors have arbitrary scale and sign:
each mode is shown normalised (its largest temperature component 1, Zs/Zb by their own largest). A complex mode: its
real part and its imaginary part (the oscillation a quarter period later) in two panels at once, on one scale.

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
    sbe = z["sigma_be"] if "sigma_be" in z.files else None
    res = z["residual"] if "residual" in z.files else None
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
    # a complex mode: its real part (the oscillation at one phase) and its imaginary part (a quarter period later)
    # side by side, both scaled by the same factor (the largest |v|), so their sizes compare; a real mode: the real
    # panel only, the imaginary panel empty
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.38, 0.38, 0.24], vertical_spacing=0.05,
                        subplot_titles=("temperature part: REAL", "temperature part: IMAGINARY (a quarter period on)",
                                        "Zs and Zb parts: real solid, imaginary dashed"))
    titles, per = [], []
    for i in range(n):
        vc = V[:, i].astype(complex)
        if imag:
            vc = -1j * vc                        # --imag: the imaginary part in the top panel
        cplx = bool(np.any(vc.imag)) and abs(complex(mu[i]).imag) > 0
        vT = vc[sl[bT.name]] / bT.scale
        a = np.nanmax(np.abs(vT[keepT])) or 1.0
        tr = []
        for row, comp in ((1, vT.real), (2, vT.imag)):
            if row == 2 and not cplx:
                continue
            xg, zg, G = binned(xT[keepT], zT[keepT], comp[keepT] / a, nz, nx)
            tr.append((row, go.Heatmap(x=xg, y=zg, z=G, colorscale="RdBu_r", zmin=-1, zmax=1, zsmooth=False,
                                       visible=(i == 0), showscale=(row == 1),
                                       colorbar=dict(title="T part", len=0.75, y=0.62), hoverongaps=False,
                                       hovertemplate="x %{x:.1f} km<br>z %{y:.0f} m<br>%{z:.3f}<extra></extra>")))
        for name, col in (("zs", "black"), ("zb", "#8e44ad")):
            p = prof(name, vc)
            if p is None:
                continue
            xs, vv = p
            if xwin:
                m = (xs >= xwin[0]) & (xs <= xwin[1])
                xs, vv = xs[m], vv[m]
            s_ = np.nanmax(np.abs(vv)) or 1.0
            tr.append((3, go.Scatter(x=xs, y=vv.real / s_, mode="lines", name=f"{name.capitalize()} real",
                                     visible=(i == 0), line=dict(color=col, width=1.5), showlegend=(i == 0))))
            if cplx:
                tr.append((3, go.Scatter(x=xs, y=vv.imag / s_, mode="lines", name=f"{name.capitalize()} imag",
                                         visible=(i == 0), line=dict(color=col, width=1.5, dash="dash"),
                                         showlegend=(i == 0))))
        per.append(tr)
        sg = sigma[i]
        osc = f", period {2 * np.pi / abs(sg.imag) / 1e3:,.2f} kyr" if abs(sg.imag) > 1e-12 else ""
        m_ = complex(mu[i])
        mus = f"{m_.real:.6f}" + (f" {'+' if m_.imag >= 0 else '-'} {abs(m_.imag):.6f}i" if abs(m_.imag) > 0 else "")
        tsc = lambda r: (f"<b style='color:#d62728'>GROWING x e in {1 / r / 1e3:,.1f} kyr</b>" if r > 0 else
                         f"<b>decaying x 1/e in {-1 / r / 1e3:,.1f} kyr</b>" if r < 0 else "<b>neutral</b>")
        titles.append(f"mode {i}: mu = {mus} (|mu| {abs(m_):.6f} over {T:g} yr): {tsc(sg.real)}{osc}"
                      + f"<br>sigma {sg.real:+.3e}/yr"
                      + (f"; BE-corrected {complex(sbe[i]).real:+.3e}/yr: {tsc(complex(sbe[i]).real)}" if sbe is not None else "")
                      + (f", residual {float(res[i]):.1e}" if res is not None else "")
                      + ("" if conv[i] else " (NOT converged)"))
    idx = []
    for tr in per:
        k0 = len(fig.data)
        for row, t in tr:
            fig.add_trace(t, row=row, col=1)
        idx.append(list(range(k0, len(fig.data))))
    steps = []
    for i in range(n):
        vis = [False] * len(fig.data)
        for k in idx[i]:
            vis[k] = True
        steps.append(dict(method="update", label=f"{i}", args=[{"visible": vis}, {"title": titles[i]}]))
    fig.update_layout(title=titles[0], template="plotly_white", height=1100, margin=dict(t=120),
                      sliders=[dict(active=0, currentvalue=dict(prefix="mode "), steps=steps, pad=dict(t=40))])
    fig.update_yaxes(title_text="z (m)", row=1, col=1)
    fig.update_yaxes(title_text="z (m)", row=2, col=1)
    fig.update_yaxes(title_text="part", row=3, col=1)
    fig.update_xaxes(title_text="x (km)", row=3, col=1)
    out = out or os.path.join(os.path.dirname(args[0]), "modes.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
