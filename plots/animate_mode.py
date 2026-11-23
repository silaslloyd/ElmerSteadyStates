#!/usr/bin/env python3
"""A flowline stability mode in time: the linear perturbation delta(t) = Re[v e^(sigma t)] of mode MODE of an
`ess stability` result (stability.npz; v the complex eigenvector, sigma = ln(mu) / T its continuous rate), over
--periods oscillation periods (--frames frames), with a time slider and a play button.

Panels: (1) the steady surface Zs and base Zb with the mode added, exaggerated so its largest surface change is --amp m
(the steady ones dashed); (2) the surface and base perturbations themselves; (3) the mode's ice temperature part over
the section. Without --grow the amplitude is held constant (the shape's motion only); with --grow it changes by
e^(Re sigma t) (Re sigma as computed: for a near-neutral pair the time stepping's damping is in it).

    python3 plots/animate_mode.py work/<case>/stability/<name>/stability.npz [MODE 2] [--periods 3] [--frames 72]
                                  [--amp 50] [--grow] [--nz 50] [--nx 400] [--x XMIN XMAX] [--out FILE.html]
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
    args, periods = opt(args, "--periods", default=3.0)
    args, nfr = opt(args, "--frames", conv=int, default=72)
    args, amp = opt(args, "--amp", default=50.0)
    args, nz = opt(args, "--nz", conv=int, default=50)
    args, nx = opt(args, "--nx", conv=int, default=400)
    args, xwin = opt(args, "--x", n=2)
    args, out = opt(args, "--out", conv=str)
    grow = "--grow" in args
    args = [a for a in args if a != "--grow"]
    if not args:
        raise SystemExit(__doc__)
    path = args[0]
    mode = int(args[1]) if len(args) > 1 else 2
    z = np.load(path)
    lay = StateLayout.from_npz(z)
    mu, sigma, V = z["mu"], z["sigma"], z["vectors"]
    T = float(z["dt"]) * int(z["n_steps"])
    s = complex(sigma[mode])
    v = V[:, mode].astype(complex)
    if abs(s.imag) < 1e-15:
        raise SystemExit(f"mode {mode} is real (mu {mu[mode]}): nothing oscillates; pick a complex one")
    period = 2 * np.pi / abs(s.imag)
    rec = lambda pre: {k[len(pre):]: (z[k], z[k.replace("gids", "vals")]) for k in z.files if k.startswith(pre)}
    diag, frozen = rec("diag_gids::"), rec("frozen_gids::")
    look = lambda d, name: next(val for k, val in d.items() if k.lower() == name)
    gx, X = look(diag, "coordinate 1")
    gz, Zc = look(diag, "coordinate 3")
    gb, B = look(frozen, "bedrock")
    xo, zo, bo = (dict(zip(g.tolist(), val)) for g, val in ((gx, X), (gz, Zc), (gb, B)))
    sl = lay.slices()
    blk = {b.name.lower(): b for b in lay.blocks}
    x0 = z["x"]                                   # the steady state (physical)

    def prof(name, vec):
        b = blk[name]
        xs = np.array([xo.get(g, np.nan) for g in b.gids]) / 1e3
        vv = vec[sl[b.name]]
        k = np.isfinite(xs)
        o = np.argsort(xs[k])
        xs, vv = xs[k][o], vv[k][o]
        xu, iu = np.unique(np.round(xs, 3), return_index=True)
        if xwin:
            m = (xu >= xwin[0]) & (xu <= xwin[1])
            xu, iu = xu[m], iu[m]
        return xu, vv[iu]

    xs_, zs0 = prof("zs", x0)
    xb_, zb0 = prof("zb", x0)
    _, vzs = prof("zs", v)
    _, vzb = prof("zb", v)
    bT = blk["temperature"]
    xT = np.array([xo.get(g, np.nan) for g in bT.gids]) / 1e3
    zT = np.array([zo.get(g, np.nan) for g in bT.gids])
    ice = np.array([zo.get(g, np.nan) >= bo.get(g, np.nan) - 0.5 for g in bT.gids]) & np.isfinite(xT)
    if xwin:
        ice &= (xT >= xwin[0]) & (xT <= xwin[1])
    vT = v[sl[bT.name]][ice]
    # the phase that puts the largest surface change at t = 0 in the real part; the scale: --amp m of surface
    j = int(np.argmax(np.abs(vzs)))
    ph = np.exp(-1j * np.angle(vzs[j]))
    vzs, vzb, vT = vzs * ph, vzb * ph, vT * ph
    sc = amp / max(np.abs(vzs).max(), 1e-300)
    tsc = 1.0 / max(np.abs(vT).max(), 1e-300)     # temperature part: largest 1
    ts = np.linspace(0.0, periods * period, nfr + 1)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.4, 0.2, 0.4],
                        subplot_titles=(f"surface and base + mode (largest surface change {amp:g} m)",
                                        "surface and base perturbation (m, at that scale)",
                                        "the mode's ice temperature part (normalised)"))

    def frame_data(t):
        e = np.exp(s * t) if grow else np.exp(1j * s.imag * t)
        dzs, dzb = (sc * vzs * e).real, (sc * vzb * e).real
        dT = (tsc * vT * e).real
        xg, zg, G = binned(xT[ice], zT[ice], dT, nz, nx)
        return [go.Scatter(x=xs_, y=zs0 + dzs, mode="lines", line=dict(color="black", width=2), name="Zs + mode"),
                go.Scatter(x=xb_, y=zb0 + dzb, mode="lines", line=dict(color="#8e44ad", width=2), name="Zb + mode"),
                go.Scatter(x=xs_, y=dzs, mode="lines", line=dict(color="black", width=1.5), name="dZs", showlegend=False),
                go.Scatter(x=xb_, y=dzb, mode="lines", line=dict(color="#8e44ad", width=1.5), name="dZb", showlegend=False),
                go.Heatmap(x=xg, y=zg, z=G, colorscale="RdBu_r", zmin=-1, zmax=1, zsmooth=False, showscale=True,
                           colorbar=dict(title="T part", len=0.38, y=0.19), hoverongaps=False)]

    d0 = frame_data(0.0)
    fig.add_trace(go.Scatter(x=xs_, y=zs0, mode="lines", line=dict(color="grey", width=1, dash="dash"),
                             name="steady Zs"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xb_, y=zb0, mode="lines", line=dict(color="grey", width=1, dash="dash"),
                             name="steady Zb"), row=1, col=1)
    rows = [1, 1, 2, 2, 3]
    for tr, r in zip(d0, rows):
        fig.add_trace(tr, row=r, col=1)
    idx = list(range(2, 2 + len(d0)))
    frames = [go.Frame(data=frame_data(t), traces=idx, name=f"{t / 1e3:.2f}") for t in ts]
    fig.frames = frames
    lim = 1.1 * max(np.abs(sc * vzs).max(), np.abs(sc * vzb).max()) * (np.exp(s.real * ts[-1]) if grow and s.real > 0 else 1)
    fig.update_yaxes(range=[-lim, lim], row=2, col=1)
    fig.update_yaxes(title_text="z (m)", row=1, col=1)
    fig.update_yaxes(title_text="m", row=2, col=1)
    fig.update_yaxes(title_text="z (m)", row=3, col=1)
    fig.update_xaxes(title_text="x (km)", row=3, col=1)
    steps = [dict(method="animate", label=f"{t / 1e3:.1f}",
                  args=[[f"{t / 1e3:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                 transition=dict(duration=0))]) for t in ts]
    ef = (f"grows x e in {1 / s.real:,.0f} yr" if s.real > 0 else f"decays x 1/e in {-1 / s.real:,.0f} yr")
    fig.update_layout(
        title=f"mode {mode}: mu {mu[mode]:.4f} over {T:g} yr; period {period:,.0f} yr, {ef}"
              + ("" if grow else " (amplitude held constant)"),
        template="plotly_white", height=1000,
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.06, xanchor="left", yanchor="top",
                          buttons=[dict(label="play", method="animate",
                                        args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, currentvalue=dict(prefix="t = ", suffix=" kyr"), pad=dict(t=60), steps=steps)])
    out = out or os.path.join(os.path.dirname(path), f"mode{mode}_animation.html")
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    print(f"-> {out} (period {period:,.0f} yr, {len(ts)} frames over {ts[-1] / 1e3:.1f} kyr)")


if __name__ == "__main__":
    main()
