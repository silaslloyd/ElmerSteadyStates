#!/usr/bin/env python3
"""The spreading test (configs/spread_ice.toml, branch test/lava-spreading): the ice thickness of the chunk states of
`ess run` (work/spread_ice/static_chunks/points/ and run_chunks/points/state_*.npz) as one plotly page:
  * a map of the thickness above the initial layer, H - H0, with a slider over the states;
  * the radial profiles H(r) about the source centre (azimuthal means), one line per state;
  * the front radius (where H - H0 first falls below --dh m, default 10, going outwards) against time, log-log, with
    the t^(1/2) slope of a constant-flux axisymmetric viscous gravity current (Huppert 1982) for comparison.

    python3 plots/spread_plot.py STATE.npz [STATE2.npz ...] [--config configs/spread_ice.toml] [--dh 10] [--out FILE.html]
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
    from scipy.interpolate import griddata
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
        t = float(s.meta.get("time", np.nan))
        r = np.hypot(x - xc, y - yc)
        edges = np.arange(0, r.max() + 5e3, 5e3)
        k = np.digitize(r, edges) - 1
        rb = 0.5 * (edges[1:] + edges[:-1])
        Hr = np.array([H[k == i].mean() if np.any(k == i) else np.nan for i in range(len(rb))])
        above = np.where(Hr - H0 < dh)[0]
        rf = float(rb[above[0]]) if len(above) else np.nan
        rows.append(dict(t=t, x=x, y=y, H=H, rb=rb, Hr=Hr, rf=rf, name=os.path.basename(p)))
        print(f"{os.path.basename(p)}: t {t:,.0f} yr, H {H.min():.1f} .. {H.max():.1f} m, front at r = {rf / 1e3:.1f} km")
    rows.sort(key=lambda r_: r_["t"])               # by model time (the static and the spreading phases' files)
    xg = np.linspace(0, L, 150)
    yg = np.linspace(0, W, 150)
    XG, YG = np.meshgrid(xg, yg)
    hmax = max(float(np.nanmax(r_["H"])) for r_ in rows) - H0
    fig = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2}, {}], [None, {}]], column_widths=[0.5, 0.5],
                        subplot_titles=("H - H0 (m)", "azimuthal mean thickness H(r)",
                                        f"front radius (H - H0 < {dh:g} m) against time"))
    for i, r_ in enumerate(rows):
        G = griddata((r_["x"], r_["y"]), r_["H"] - H0, (XG, YG), method="linear")
        fig.add_trace(go.Heatmap(x=xg / 1e3, y=yg / 1e3, z=G, colorscale="Viridis", zmin=0, zmax=hmax,
                                 visible=(i == len(rows) - 1), colorbar=dict(title="m", x=0.46),
                                 hovertemplate="x %{x:.0f} y %{y:.0f} km<br>H - H0 %{z:.1f} m<extra></extra>"),
                      row=1, col=1)
    nmap = len(rows)
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
    fig.update_xaxes(title_text="x (km)", row=1, col=1, scaleanchor="y", constrain="domain")
    fig.update_yaxes(title_text="y (km)", row=1, col=1)
    fig.update_xaxes(title_text="r (km)", row=1, col=2)
    fig.update_yaxes(title_text="H (m)", row=1, col=2)
    fig.update_xaxes(title_text="t (yr)", type="log", row=2, col=2)
    fig.update_yaxes(title_text="front radius (km)", type="log", row=2, col=2)
    steps = [dict(label=f"{r_['t']:,.0f}", method="restyle",
                  args=[{"visible": [j == i for j in range(nmap)]}, list(range(nmap))]) for i, r_ in enumerate(rows)]
    fig.update_layout(template="plotly_white", height=850, title="spreading test: cold viscous gravity current",
                      sliders=[dict(active=nmap - 1, currentvalue=dict(prefix="map at t (yr) = "), steps=steps,
                                    x=0.0, len=0.48, pad=dict(t=40))],
                      legend=dict(x=1.02, y=1.0))
    out = out or os.path.join(os.path.dirname(os.path.abspath(sorted(args)[-1])), "spread.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
