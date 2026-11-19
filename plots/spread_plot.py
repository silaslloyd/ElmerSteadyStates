#!/usr/bin/env python3
"""The spreading test (configs/spread_ice.toml, branch test/lava-spreading): the ice thickness of the chunk states of
`ess run` (work/spread_ice/static_chunks/points/ and run_chunks/points/state_*.npz) as one plotly page:
  * the surface Zs in 3D on the model's footprint nodes, coloured by the surface horizontal speed sqrt(u^2 + v^2)
    (log10, from the Stokes solution), with a slider over the states; the window: the current's extent (H - H0 >
    --dh) over all states, padded by half; --zaspect (default 0.35) the box's height / width;
  * the radial profiles H(r) about the source centre (azimuthal means), one line per state;
  * the front radius (where H - H0 first falls below --dh m, default 10, going outwards) against time, log-log, with
    the t^(1/2) slope of a constant-flux axisymmetric viscous gravity current (Huppert 1982) for comparison.

    python3 plots/spread_plot.py STATE.npz [STATE2.npz ...] [--config configs/spread_ice.toml] [--dh 10] [--zaspect 0.35] [--out FILE.html]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.statefile import find_name, load_state  # noqa: E402


def arg(args, name, default, conv=str):
    if name in args:
        k = args.index(name)
        v = conv(args[k + 1])
        del args[k:k + 2]
        return v
    return default


def main():
    args = sys.argv[1:]
    cfgf = arg(args, "--config", os.path.join(ROOT, "configs", "spread_ice.toml"))
    dh = arg(args, "--dh", 10.0, float)
    zaspect = arg(args, "--zaspect", 0.35, float)   # the 3D box's height / width (the vertical exaggeration follows)
    out = arg(args, "--out", None)
    if not args:
        raise SystemExit(__doc__)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    c = tomllib.load(open(cfgf, "rb"))["case"]
    L, W = float(c["spread_length_km"]) * 1e3, float(c["width"])
    xc, yc = float(c.get("source_x_km", L / 2e3)) * 1e3, float(c.get("source_y_km", W / 2e3)) * 1e3
    H0 = float(c.get("spread_thickness", 100.0))
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = []
    for p in sorted(args):
        s = load_state(p)
        allc = {**s.diag, **s.frozen}
        cx, cy = allc[find_name(allc, "coordinate 1")], allc[find_name(allc, "coordinate 2")]
        X, Y = dict(zip(cx[0].tolist(), cx[1])), dict(zip(cy[0].tolist(), cy[1]))
        blk = {b.name.lower(): b for b in s.layout.blocks}
        sl = s.layout.slices()
        gs = blk["zs"].gids
        zs = s.x[sl[blk["zs"].name]]
        B = dict(zip(*allc[find_name(allc, "bedrock")]))
        x = np.array([X[g] for g in gs])
        y = np.array([Y[g] for g in gs])
        H = zs - np.array([B[g] for g in gs])
        # the horizontal speed at the surface nodes (m/yr), from the Stokes solution (NaN if the state has none)
        try:
            V = [dict(zip(*allc[find_name(allc, f"velocity {k}")])) for k in (1, 2)]
            U = np.hypot(*[np.array([v_.get(g, np.nan) for g in gs], float) for v_ in V])
        except KeyError:
            U = np.full(len(gs), np.nan)
        t = float(s.meta.get("time", np.nan))
        rows.append(dict(t=t, x=x, y=y, zs=zs, H=H, U=U, r=np.hypot(x - xc, y - yc), name=os.path.basename(p)))
    rows.sort(key=lambda r_: r_["t"])               # by model time (the static and the spreading phases' files)
    # the window: the current's extent over all states (H - H0 > dh), padded by half, at least 3 km about the source
    rmax = max([float(r_["r"][r_["H"] - H0 > dh].max()) for r_ in rows if np.any(r_["H"] - H0 > dh)] + [0.0])
    half = max(1.5 * rmax, 3e3)
    # radial profiles (azimuthal means in bins of ~half/60) and the front radius
    dr = float(np.clip(half / 60, 50.0, 5e3))
    for r_ in rows:
        edges = np.arange(0, r_["r"].max() + dr, dr)
        k = np.digitize(r_["r"], edges) - 1
        r_["rb"] = 0.5 * (edges[1:] + edges[:-1])
        r_["Hr"] = np.array([r_["H"][k == i].mean() if np.any(k == i) else np.nan for i in range(len(r_["rb"]))])
        above = np.where(r_["Hr"] - H0 < dh)[0]
        r_["rf"] = float(r_["rb"][above[0]]) if len(above) else np.nan
        print(f"{r_['name']}: t {r_['t']:,.0f} yr, H {r_['H'].min():.1f} .. {r_['H'].max():.1f} m, front at r = "
              f"{r_['rf'] / 1e3:.2f} km")
    allU = np.concatenate([r_["U"][np.isfinite(r_["U"]) & (r_["U"] > 0)] for r_ in rows] + [np.array([1e-6])])
    u_hi = float(np.log10(np.percentile(allU, 99.5)))
    u_lo = u_hi - 4.0
    hmax = max(float(np.nanmax(r_["H"])) for r_ in rows) - H0
    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2, "type": "scene"}, {}], [None, {}]],
                        column_widths=[0.55, 0.45],
                        subplot_titles=("the surface Zs, coloured by the surface horizontal speed (m/yr, log10)",
                                        "azimuthal mean thickness H(r)",
                                        f"front radius (H - H0 < {dh:g} m) against time"))
    # the surface as a 3D mesh on the model's own footprint nodes (Delaunay of the surface nodes) within the window,
    # the colour log10 of the horizontal speed at the surface
    from scipy.spatial import Delaunay
    map_idx = []
    zlo = min(float(np.nanmin(r_["zs"] - r_["H"])) for r_ in rows)
    zhi = max(float(np.nanmax(r_["zs"])) for r_ in rows)
    for i, r_ in enumerate(rows):
        m = (np.abs(r_["x"] - xc) <= half) & (np.abs(r_["y"] - yc) <= half)
        xs, ys = r_["x"][m], r_["y"][m]
        tri = Delaunay(np.c_[xs, ys]).simplices
        with np.errstate(divide="ignore", invalid="ignore"):
            lu = np.log10(r_["U"][m])
        lu = np.where(np.isfinite(lu), lu, u_lo)
        fig.add_trace(go.Mesh3d(x=(xs / 1e3).astype(np.float32), y=(ys / 1e3).astype(np.float32),
                                z=r_["zs"][m].astype(np.float32), i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                                intensity=lu.astype(np.float32), colorscale="Inferno", cmin=u_lo, cmax=u_hi,
                                colorbar=dict(title="log10 u_h (m/yr)", x=0.5, len=0.8), flatshading=False,
                                lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1, roughness=0.9),
                                customdata=np.c_[(r_["H"][m] - H0)].astype(np.float32),
                                hovertemplate="x %{x:.2f} y %{y:.2f} km<br>Zs %{z:.0f} m<br>log10 u_h "
                                              "%{intensity:.2f}<extra></extra>",
                                showscale=True, visible=i == len(rows) - 1), row=1, col=1)
        map_idx.append([len(fig.data) - 1])
    allmap = [j for idx in map_idx for j in idx]
    for i, r_ in enumerate(rows):
        fig.add_trace(go.Scatter(x=r_["rb"] / 1e3, y=r_["Hr"], mode="lines", name=f"{r_['t']:,.0f} yr",
                                 line=dict(color=f"rgba(30,60,160,{0.2 + 0.8 * (i + 1) / len(rows):.2f})", width=1.4)),
                      row=1, col=2)
    tt = np.array([r_["t"] for r_ in rows])
    rf = np.array([r_["rf"] for r_ in rows])
    ok = np.isfinite(rf) & (tt > 0)
    fig.add_trace(go.Scatter(x=tt[ok], y=rf[ok] / 1e3, mode="markers+lines", name="front", showlegend=False),
                  row=2, col=2)
    if ok.sum() >= 1:
        t1, r1 = tt[ok][-1], rf[ok][-1] / 1e3
        tl = np.array([tt[ok].min(), tt[ok].max()])
        fig.add_trace(go.Scatter(x=tl, y=r1 * (tl / t1) ** 0.5, mode="lines", name="t^1/2", showlegend=False,
                                 line=dict(dash="dot", color="grey")), row=2, col=2)
    vex = zaspect * (2 * half) / max(zhi - zlo, 1.0)
    fig.update_scenes(xaxis=dict(title="x (km)", range=[(xc - half) / 1e3, (xc + half) / 1e3]),
                      yaxis=dict(title="y (km)", range=[(yc - half) / 1e3, (yc + half) / 1e3]),
                      zaxis=dict(title="z (m)", range=[zlo, zhi * 1.02]),
                      aspectmode="manual", aspectratio=dict(x=1, y=1, z=zaspect),
                      camera=dict(eye=dict(x=1.3, y=-1.3, z=0.9)), row=1, col=1)
    fig.update_xaxes(title_text="r (km)", row=1, col=2)
    fig.update_yaxes(title_text="H (m)", row=1, col=2)
    fig.update_xaxes(title_text="t (yr)", type="log", row=2, col=2)
    fig.update_yaxes(title_text="front radius (km)", type="log", row=2, col=2)
    steps = [dict(label=f"{r_['t']:,.0f}", method="restyle",
                  args=[{"visible": [j in set(map_idx[i]) for j in allmap]}, allmap]) for i, r_ in enumerate(rows)]
    fig.update_layout(template="plotly_white", height=850, title=f"spreading test (surface: vertical exaggeration {vex:.1f})",
                      sliders=[dict(active=len(rows) - 1, currentvalue=dict(prefix="surface at t (yr) = "), steps=steps,
                                    x=0.0, len=0.48, pad=dict(t=40))],
                      legend=dict(x=1.02, y=1.0))
    out = out or os.path.join(os.path.dirname(os.path.abspath(sorted(args)[-1])), "spread.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
