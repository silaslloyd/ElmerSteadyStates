#!/usr/bin/env python3
"""The steady water sheet (the CS sheet of [elmer] sheet_solver = "cs", as ess/spectrum.py hydrology_cs) on the temperate
patch of flowline steady states, integrated from the grounding line upstream; one page, one line per state.

On the grounded temperate bed, from the patch head x_h to the grounding line x_g (melt and Phi0 binned to --dx m):
    q(x) = int_{x_h}^{x} m dx'                        (no water enters at the cold/temperate boundary)
    q = -K(N(h)) (Phi0' - N'(h) h')                   (flux down the water potential Phi0 - N)
    N(h) = N0 hr / sqrt(h^2 + eps^2),   K(N) = D0 (s(N)^-k0 - s(N(0))^-k0),   s(N) = ln(1 + e^(10 N)) / 10
    h(x_g) = hr                                       (integrated upstream from the grounding line:
                                                       dh/dx = (Phi0' + q / K) / N'(h))
m: the melt rate (Temperature Loads, as plots/plot_melt.py), Phi0 = Stokes pressure + rho_w g z_b at the bed (MPa).
Status: ok; dry (h reaches 0 before the head: the patch cannot carry its water upstream); flood (h > h_max).

    python3 plots/sweep_hydrology.py STATE.npz [STATE2.npz ...] [--config configs/flowline_rect_coarse.toml]
                                     [--D0 1e7] [--N0 1] [--hr 0.1] [--eps 0.001] [--k0 1.3333] [--h-max 1000]
                                     [--dx 250] [--x XMIN XMAX] [--h-log] [--verbose] [--out FILE.html]

Parameters: Marine_Term's (elmer/ess_hydro.lua), the config's [hydrology] overriding, the options overriding both.
Default page: the first state's name + _hydro.html. Panels: melt rate, flux q, sheet thickness h (log), N.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from ess.case import eval_lua_parameters  # noqa: E402
from ess.spectrum import _bed, cs_N, cs_sheet_params, patch_hydrology_cs  # noqa: E402
from ess.statefile import find_name, load_state  # noqa: E402


def opt(args, name, n=1, conv=float, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = [conv(a) for a in args[k + 1:k + 1 + n]]
    return args[:k] + args[k + 1 + n:], (v[0] if n == 1 else v)


def main():
    args = sys.argv[1:]
    args, cfgf = opt(args, "--config", conv=str, default=os.path.join(ROOT, "configs", "flowline_rect_coarse.toml"))
    over = {}
    for o, k in (("--D0", "D0"), ("--N0", "N0"), ("--hr", "hr"), ("--eps", "eps"), ("--k0", "k0"), ("--h-max", "h_max")):
        args, v = opt(args, o)
        if v is not None:
            over[k] = v
    args, dx = opt(args, "--dx", default=250.0)
    hlog = "--h-log" in args                      # the sheet thickness axis log at first (default linear; buttons switch)
    args = [a for a in args if a != "--h-log"]
    verbose = "--verbose" in args                 # how each integration ended, and the first bins' terms
    args = [a for a in args if a != "--verbose"]
    args, xwin = opt(args, "--x", n=2)
    args, out = opt(args, "--out", conv=str)
    if not args:
        raise SystemExit(__doc__)
    try:
        try:
            import tomllib
        except ImportError:                      # python < 3.11
            import tomli as tomllib
        with open(cfgf, "rb") as f:
            cfg = tomllib.load(f)
    except Exception as e:                       # noqa: BLE001
        print(f"config {cfgf} not read ({e}): Marine_Term's sheet parameters")
        cfg = {}
    hp = {**cs_sheet_params(cfg), **over}
    pf = os.path.join(ROOT, (cfg.get("case", {}).get("parameters") or "case/parameters_fine2x.lua"))
    P = eval_lua_parameters(pf)
    print("sheet: " + ", ".join(f"{k} {v:g}" for k, v in hp.items()) + f"; rho_i, L, rho_w, g from {pf}")
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    from plotly.subplots import make_subplots
    res = []
    sys.path.insert(0, HERE)
    from state_pick import prefer_fine
    for path in prefer_fine(args):          # one state per sweep length d: the finest mesh's
        s = load_state(path)
        g, xpos, grounded = _bed(s.frozen, s.diag)
        ct = s.frozen[find_name(s.frozen, "ColdTempMask")]
        ctv = dict(zip(ct[0].tolist(), ct[1]))
        temp = np.array([ctv.get(k, -1.0) > 0 for k in g.tolist()])
        if not (grounded & temp).any():
            print(f"{path}: no grounded temperate bed")
            continue
        xg = float(xpos[grounded].max())
        d = xg - float(xpos[grounded & temp].min())          # the patch: from its head to the grounding line
        if verbose:
            print(f"{path}:")
        ok, st, row, hyd = patch_hydrology_cs(s.diag, s.layout, s.x, s.frozen, xg, d, {**hp, "verbose": verbose},
                                              rhoi=P["rhoi"], Lw=P["Lw"], rhow=P["rhow"], gravity=P["gravity"], dx_bin=dx)
        if st == "no-diag":
            print(f"{path}: {row['note']}: skipped")
            continue
        # back to the bins along x (the node fields of patch_hydrology_cs, averaged per x)
        gh, hv = hyd["Water Sheet Thickness"]
        _, qv = hyd["Water Flux"]
        _, mv = hyd["Melt Rate"]
        idx = {k: i for i, k in enumerate(g.tolist())}
        xs = xpos[[idx[k] for k in gh.tolist()]] / 1e3
        sel = np.isfinite(qv)
        ux = np.unique(np.round(xs[sel], 3))
        col = lambda v: np.array([np.nanmean(v[sel][np.round(xs[sel], 3) == u]) if np.isfinite(
            v[sel][np.round(xs[sel], 3) == u]).any() else np.nan for u in ux])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hc, qc, mc = col(hv), col(qv), col(mv)
        Nc = np.array([cs_N(h, hp) if np.isfinite(h) else np.nan for h in hc])
        lab0 = (f"d = {float(s.meta['sweep_d']) / 1e3:.1f} km"
                + (f", it {int(s.meta['newton_it'])}" if "newton_it" in s.meta else "")
                if "sweep_d" in s.meta else os.path.basename(path))
        lab = f"{lab0} ({st})"
        hmin_x = float(ux[np.isfinite(hc)].min()) if np.isfinite(hc).any() else xg / 1e3
        print(f"{lab0}: patch {(xg - d) / 1e3:.2f} .. {xg / 1e3:.2f} km, status {st}: h at the head {row['h_head']:.4g} m, "
              f"h min {row['h_min']:.4g} m, q at the grounding line {row['q_gl']:.4g} m^2/yr, q min {row['q_min']:.4g}"
              + (f"; dry upstream of x = {hmin_x:.2f} km" if st == "dry" else ""))
        res.append(dict(d=s.meta.get("sweep_d"), lab=lab, x=ux, h=hc, q=qc, m=mc, N=Nc, st=st, xdry=hmin_x, xg=xg / 1e3))
    if not res:
        raise SystemExit("no state with a temperate patch and the diagnostics")
    res.sort(key=lambda r: (r["d"] is not None, -(float(r["d"]) if r["d"] is not None else 0)))
    cols = sample_colorscale("Plasma", [0.05 + 0.85 * k / max(len(res) - 1, 1) for k in range(len(res))])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=("melt rate m (m/yr; < 0 refreezing)", "water flux q = int m dx from the patch head (m^2/yr)",
                                        "sheet thickness h (m), h = hr at the grounding line; x: dry", "effective pressure N (MPa)"))
    for r, c in zip(res, cols):
        sel = np.ones(len(r["x"]), bool) if not xwin else (r["x"] >= xwin[0]) & (r["x"] <= xwin[1])
        for row_, key, show in ((1, "m", True), (2, "q", False), (3, "h", False), (4, "N", False)):
            fig.add_trace(go.Scatter(x=r["x"][sel], y=r[key][sel], mode="lines", name=r["lab"], legendgroup=r["lab"],
                                     showlegend=show, line=dict(color=c, width=1.5),
                                     hovertemplate=f"{r['lab']}<br>x %{{x:.2f}} km<br>{key} %{{y:.4g}}<extra></extra>"),
                          row=row_, col=1)
        if r["st"] == "dry":
            hd = r["h"][np.isfinite(r["h"])]
            fig.add_trace(go.Scatter(x=[r["xdry"]], y=[hd[0] if hd.size else hp["hr"]], mode="markers", legendgroup=r["lab"],
                                     showlegend=False, marker=dict(symbol="x", size=10, color=c, line=dict(width=1)),
                                     hovertemplate=f"{r['lab']}: dry at x %{{x:.2f}} km<extra></extra>"), row=3, col=1)
    fig.add_hline(y=0, line=dict(color="grey", width=0.8), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="grey", width=0.8), row=2, col=1)
    fig.update_yaxes(type="log" if hlog else "linear", row=3, col=1)
    fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=1.0, xanchor="right", y=0.49, yanchor="bottom",
                                        buttons=[dict(label="h linear", method="relayout", args=[{"yaxis3.type": "linear"}]),
                                                 dict(label="h log", method="relayout", args=[{"yaxis3.type": "log"}])])])
    fig.update_xaxes(title_text="x (km)", row=4, col=1)
    if xwin:
        fig.update_xaxes(range=xwin)
    fig.update_layout(title="steady CS water sheet on the temperate patch, integrated from the grounding line upstream ("
                            + ", ".join(f"{k} {v:g}" for k, v in hp.items()) + ")",
                      template="plotly_white", height=1200, hovermode="closest",
                      legend=dict(title="state (status)"))
    out = out or os.path.splitext(args[0])[0] + "_hydro.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
