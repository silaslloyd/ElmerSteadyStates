#!/usr/bin/env python3
"""
One eigenmode of a 3D-strip stability analysis (`ess stability --spanwise`): maps over x and y of the
basal-temperature and surface perturbations, the across-flow spectrum of every mode, and the leading
modes' growth rates against spanwise wavelength.

Press Run in an IDE (settings below), or:
    python plots/plot_mode_3d.py work/land_margin_3d/stability/<label>_spanwise/stability.npz 0
Needs numpy and plotly. Eigenvectors have arbitrary scale and sign; complex modes show the real part.
"""
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from ess.statefile import StateLayout  # noqa: E402

# ---- settings for pressing Run (paths relative to the repository)
FILE = "work/land_margin_3d/stability/steady_p268.228_spanwise/stability.npz"
MODE = 0
PART = "real"        # "real" or "imag" (complex modes)
SHOW = True


def records(z, prefix):
    return {k.split("::", 1)[1]: (z[k], z[prefix + "vals::" + k.split("::", 1)[1]])
            for k in z.files if k.startswith(prefix + "gids::")}


def look(rec, name, g):
    key = next(k for k in rec if k.lower() == name)
    gc, vc = rec[key]
    return vc[np.searchsorted(gc, g)]


def grid(x, y, v):
    """Values on the structured (x, y) node grid of the strip: unique x by unique y."""
    xk, yk = np.round(x, 0), np.round(y, 0)
    ux, uy = np.unique(xk), np.unique(yk)
    G = np.full((len(uy), len(ux)), np.nan)
    G[np.searchsorted(uy, yk), np.searchsorted(ux, xk)] = v
    return ux, uy, G


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else FILE
    f = f if os.path.isabs(f) else os.path.join(REPO, f)
    mode = int(sys.argv[2]) if len(sys.argv) > 2 else MODE
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    z = np.load(f, allow_pickle=False)
    lay = StateLayout.from_npz(z)
    vec = z["vectors"][:, mode]
    vec = vec.real if PART == "real" else vec.imag
    split = lay.split(vec)
    gids = {b.name: b.gids for b in lay.blocks}
    diag, frozen = records(z, "diag_"), records(z, "frozen_")
    coords = {**diag}
    tname = next(n for n in lay.names if n.lower() == "temperature")
    zname = next(n for n in lay.names if n.lower() == "zs")

    # bed nodes: those carrying the frozen cold/temperate mask
    gb = frozen[next(k for k in frozen if k.lower() == "coldtempmask")][0]
    Tb = split[tname][np.searchsorted(gids[tname], gb)]
    xb, yb = look(coords, "coordinate 1", gb), look(coords, "coordinate 2", gb)
    gs = gids[zname]
    xs, ys = look(coords, "coordinate 1", gs), look(coords, "coordinate 2", gs)

    mu, sigma = z["mu"][mode], z["sigma"][mode]
    has_span = "spanwise_n" in z.files
    head = (f"mode {mode}: |mu| = {abs(mu):.5f}, sigma = {sigma.real:+.3e} /yr"
            + (f", dominant n = {z['spanwise_n'][mode]} (wavelength {z['spanwise_wavelength'][mode] / 1e3:.3g} km)"
               if has_span else "") + ("" if z["converged"][mode] else "  [not converged]")
            + f"  -  Tsurf = {float(z['p']):.4f} K")

    rows = 4 if has_span else 2
    titles = ["basal temperature perturbation (K, arbitrary scale)", "surface perturbation (m, arbitrary scale)"]
    if has_span:
        titles += ["across-flow spectrum of each mode (energy fraction in n)",
                   "leading modes: growth rate against spanwise wavelength"]
    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles, vertical_spacing=0.07)
    for r, (x, y, v, unit) in enumerate([(xb, yb, Tb, "K"), (xs, ys, split[zname], "m")], start=1):
        ux, uy, G = grid(x, y, v)
        lim = np.nanmax(np.abs(G)) or 1.0
        fig.add_trace(go.Heatmap(x=ux / 1e3, y=uy / 1e3, z=G, colorscale="RdBu_r", zmin=-lim, zmax=lim,
                                 colorbar=dict(len=0.2, y=1 - (r - 0.5) / rows, title=unit),
                                 hovertemplate="x %{x:.1f} km<br>y %{y:.2f} km<br>%{z:.3g}<extra></extra>"), r, 1)
        fig.update_yaxes(title_text="y (km)", row=r, col=1)
        fig.update_xaxes(title_text="x (km)", row=r, col=1)
    if has_span:
        S = z["spanwise_spectrum"]
        fig.add_trace(go.Heatmap(z=S, x=np.arange(S.shape[1]), y=np.arange(S.shape[0]), colorscale="Viridis",
                                 colorbar=dict(len=0.2, y=1 - 2.5 / rows, title="fraction"),
                                 hovertemplate="mode %{y}<br>n %{x}<br>%{z:.2f}<extra></extra>"), 3, 1)
        fig.update_xaxes(title_text="n (across-flow cos(n pi y / W))", row=3, col=1)
        fig.update_yaxes(title_text="mode", row=3, col=1)
        lam = z["spanwise_wavelength"] / 1e3
        conv = z["converged"]
        fig.add_trace(go.Scatter(x=lam, y=z["sigma"].real, mode="markers+text",
                                 text=[str(i) for i in range(len(lam))], textposition="top center",
                                 marker=dict(size=9, color=np.where(conv, "#1f77b4", "#bbbbbb")),
                                 hovertemplate="wavelength %{x:.3g} km<br>sigma %{y:.3e} /yr<extra></extra>"), 4, 1)
        fig.add_hline(y=0, line=dict(dash="dot", color="grey"), row=4, col=1)
        fig.update_xaxes(title_text="spanwise wavelength (km)", row=4, col=1)
        fig.update_yaxes(title_text="sigma (1/yr)", row=4, col=1)
    fig.update_layout(title=head, height=380 * rows, template="plotly_white", showlegend=False)
    out = os.path.join(os.path.dirname(f), f"mode_{mode}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
