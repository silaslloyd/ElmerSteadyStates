#!/usr/bin/env python3
"""A flowline state's temperature as a filled-contour section (plotly): x along the flow, z elevation, the ice
temperature on a regular z grid column by column (no smoothing across columns), the upper surface Zs, the ice base Zb
and the bed drawn in. The temperature is clipped to the surface temperature (the state's parameter, Tsurf; --tsurf K to set it) ..
melting (273.15 K); how many nodes were outside is printed. The
flowline mesh is one element wide: the nodes at the smallest y are used.

    python3 plots/plot_section.py STATE.npz [--x XMIN XMAX (km)] [--tmin K] [--tmax K] [--tsurf K] [--umax M/YR] [--nz 300] [--rock]
                                  [--out FILE.html]

Buttons switch between the temperature and the velocities u (along the flow; colours up to the fastest grounded
ice, --umax to set them) and w (vertical, symmetric about 0), from the state's diagnostics Velocity 1 / Velocity 3.

--rock: the bedrock's temperature too (down to the bottom of the 10 zscale rock), not only the ice.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from plot_state import column_grid, load, on_side  # noqa: E402
from ess.statefile import find_name  # noqa: E402


def opt(args, name, n=1, conv=float, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = [conv(a) for a in args[k + 1:k + 1 + n]]
    return args[:k] + args[k + 1 + n:], (v[0] if n == 1 else v)


def main():
    args = sys.argv[1:]
    args, xwin = opt(args, "--x", n=2)
    args, tmin = opt(args, "--tmin")
    args, tmax = opt(args, "--tmax")
    args, nz = opt(args, "--nz", conv=int, default=300)
    args, tsurf = opt(args, "--tsurf")
    args, umax = opt(args, "--umax")
    args, out = opt(args, "--out", conv=str)
    rock = "--rock" in args
    args = [a for a in args if a != "--rock"]
    if len(args) != 1:
        raise SystemExit(__doc__)
    path = args[0]
    import plotly.graph_objects as go

    layout, fields, diag, frozen, title, _ = load(path, 0)
    coords = {f"coordinate {c}": diag[find_name(diag, f"coordinate {c}")] for c in (1, 2, 3)}
    gx, X = coords["coordinate 1"]
    gz, Z = coords["coordinate 3"]

    def xz(g):
        return X[np.searchsorted(gx, g)] / 1e3, Z[np.searchsorted(gz, g)]

    def profile(name, src=None):
        if src is None:
            nm = next((n for n in layout.names if n.lower() == name), None)
            if nm is None:
                return None
            g = next(b.gids for b in layout.blocks if b.name == nm)
            v = fields[nm]
        else:
            g, v = src
        s = on_side(g, coords)
        x_ = xz(g[s])[0]
        o = np.argsort(x_)
        return x_[o], v[s][o]
    pzs, pzb = profile("zs"), profile("zb")
    pbed = profile(None, frozen[find_name(frozen, "bedrock")]) if any(k.lower() == "bedrock" for k in frozen) else None
    if pbed is not None:                         # the bedrock record holds every node of a column: its base level
        xb_, vb_ = pbed
        k_ = np.unique(np.round(xb_, 6), return_index=True)[1]
        pbed = (xb_[k_], vb_[k_])
    tname = next(n for n in layout.names if n.lower() == "temperature")
    gT = next(b.gids for b in layout.blocks if b.name == tname)
    s = on_side(gT, coords)
    xt, zt = xz(gT[s])
    T = fields[tname][s]
    if not rock and pzb is not None:             # the ice only: at or above the base
        ice = zt >= np.interp(xt, *pzb) - 0.5
        xt, zt, T = xt[ice], zt[ice], T[ice]
    if xwin:
        m = (xt >= xwin[0]) & (xt <= xwin[1])
        xt, zt, T = xt[m], zt[m], T[m]
    # clipped at melting above and at the surface temperature below (the state's parameter p, Tsurf; --tsurf to
    # set it); how many nodes were outside is reported
    ts = tsurf if tsurf is not None else float(np.load(path)["p"])
    above, below = int(np.sum(T > 273.15 + 1e-6)), int(np.sum(T < ts - 1e-6))
    T = np.clip(T, ts, 273.15)
    xg, zg, G = column_grid(xt, zt, T, nz)
    lo = tmin if tmin is not None else ts
    hi = tmax if tmax is not None else 273.15
    print(f"{path}: {len(xg)} columns, T clipped to {ts:.2f} .. 273.15 K ({above} plotted nodes were above melting, "
          f"{below} below the surface temperature); colours {lo:.2f} .. {hi:.2f}")
    traces = [go.Contour(x=xg, y=zg, z=G, colorscale="RdBu_r", zmin=lo, zmax=hi,
                         contours=dict(start=lo, end=hi, size=(hi - lo) / 30, coloring="fill",
                                       showlines=False),
                         colorbar=dict(title="T (K)"), connectgaps=False,
                         hovertemplate="x %{x:.1f} km<br>z %{y:.0f} m<br>T %{z:.2f} K<extra></extra>",
                         name="T")]
    # the velocities (the state's diagnostics Velocity 1 / Velocity 3, on the ice nodes), as two more fields
    fields_v = []
    for c, lab, cs in ((1, "u (along-flow velocity, m/yr)", "Viridis"), (3, "w (vertical velocity, m/yr)", "RdBu_r")):
        try:
            gv, vv = diag[find_name(diag, f"velocity {c}")]
        except KeyError:
            print(f"no Velocity {c} in the state: no {lab.split()[0]} view")
            continue
        sv = on_side(gv, coords)
        xv, zv = xz(gv[sv])
        val = vv[sv]
        if pzb is not None:
            m = zv >= np.interp(xv, *pzb) - 0.5
            xv, zv, val = xv[m], zv[m], val[m]
        if xwin:
            m = (xv >= xwin[0]) & (xv <= xwin[1])
            xv, zv, val = xv[m], zv[m], val[m]
        xg_, zg_, G_ = column_grid(xv, zv, val, nz)
        if c == 1:          # up to the fastest grounded ice (the shelf, much faster, saturates; --umax to set it)
            xgl = None
            if pzb is not None and pbed is not None:
                fl = pzb[1] - np.interp(pzb[0], *pbed) > 1.0          # the base above the bed: floating
                xgl = float(pzb[0][np.argmax(fl)]) if fl.any() else None
            gsel = (xg_ <= xgl) if xgl is not None else np.ones(len(xg_), bool)
            hi_ = umax if umax is not None else float(np.nanmax(G_[:, gsel]))
            lo_, mid = 0.0, None
        else:
            hi_ = float(np.nanpercentile(np.abs(G_), 99)) or 1e-6
            lo_, mid = -hi_, 0.0
        print(f"{lab}: {np.nanmin(G_):.4g} .. {np.nanmax(G_):.4g}; colours {lo_:.4g} .. {hi_:.4g}")
        fields_v.append(go.Contour(x=xg_, y=zg_, z=G_, colorscale=cs, zmin=lo_, zmax=hi_,
                                   contours=dict(start=lo_, end=hi_, size=(hi_ - lo_) / 30, coloring="fill",
                                                 showlines=False),
                                   colorbar=dict(title=lab.split()[0] + " (m/yr)"), connectgaps=False, visible=False,
                                   hovertemplate="x %{x:.1f} km<br>z %{y:.0f} m<br>" + lab.split()[0]
                                   + " %{z:.4g} m/yr<extra></extra>", name=lab))
    nfield = 1 + len(fields_v)
    traces += fields_v
    nline = 0
    for p_, nm, col, dash in ((pzs, "Zs (surface)", "black", None), (pzb, "Zb (ice base)", "#8e44ad", None),
                              (pbed, "bed", "#7f6000", "dot")):
        if p_ is None:
            continue
        x_, z_ = p_
        if xwin:
            m = (x_ >= xwin[0]) & (x_ <= xwin[1])
            x_, z_ = x_[m], z_[m]
        traces.append(go.Scatter(x=x_, y=z_, mode="lines", name=nm, line=dict(color=col, width=1.6, dash=dash),
                                 hovertemplate=f"x %{{x:.1f}} km<br>{nm} %{{y:.0f}} m<extra></extra>"))
        nline += 1
    fig = go.Figure(traces)
    names = ["temperature"] + [t.name.split(" (")[0] for t in fields_v]
    buttons = [dict(label=nm, method="update",
                    args=[{"visible": [j == i for j in range(nfield)] + [True] * nline},
                          {"title.text": f"{os.path.basename(path)}: {nm}, {title}"}])
               for i, nm in enumerate(names)]
    fig.update_layout(title=f"{os.path.basename(path)}: temperature, {title}", template="plotly_white",
                      xaxis=dict(title="x (km)"), yaxis=dict(title="z (m)"),
                      updatemenus=[dict(type="buttons", direction="right", buttons=buttons, x=0.0, y=1.06,
                                        xanchor="left", yanchor="bottom", showactive=True, active=0)],
                      legend=dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom", bgcolor="rgba(255,255,255,0.7)"))
    out = out or os.path.splitext(path)[0] + "_section.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
