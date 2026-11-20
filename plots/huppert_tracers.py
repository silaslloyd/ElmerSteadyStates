#!/usr/bin/env python3
"""Tracers in Huppert's (1982) analytical drop, for as long as you like: a volume V of viscous fluid spreading on a
dry, rigid, no-slip flat bed (lubrication), h(r, t) = A(t) (3/16 (1 - (r/R)^2))^(1/3), R = xi_N (D V^3 t)^(1/8),
A = xi_N^(2/3) (V / (D t))^(1/4), D = rho g / (3 eta); the velocity in the drop
    u(r, z) = 3 D (-dh/dr) (h z - z^2 / 2),   w = -(1/r) d/dr (r Q),  Q = int_0^z u dz' = 3 D (-dh/dr) (h z^2/2 - z^3/6)
Tracers start on the surface (and --depths m below it) at --seed (yr after the drop's start, with the virtual origin
t_v that puts the front at the drop's radius at t = 0, as plots/spread_drop_compare.py) and are advected (RK45) to
--tmax. Page: (1) the section r-z with the tracers, a slider over log-spaced times; (2) the tracers' paths in the
similarity coordinates (r/R, z/h), where the motion is the same at every time (only its pace slows).

    python3 plots/huppert_tracers.py [--config configs/spread_drop_f0500.toml] [--seed 200] [--tmax 1e6]
                                     [--n 60] [--depths 0,10,20] [--frames 40] [--out FILE.html]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.case import eval_lua_parameters  # noqa: E402

XI = (2 ** 10 / (3 ** 4 * np.pi ** 3)) ** 0.125
C = 3 / 16


def arg(args, name, default, conv=str):
    if name in args:
        k = args.index(name)
        v = conv(args[k + 1])
        del args[k:k + 2]
        return v
    return default


class Huppert:
    def __init__(self, V, D):
        self.V, self.D = V, D

    def R(self, t):
        return XI * (self.D * self.V ** 3 * t) ** 0.125

    def A(self, t):
        return XI ** (2 / 3) * (self.V / (self.D * t)) ** 0.25

    def h(self, r, t):
        R = self.R(t)
        return self.A(t) * np.cbrt(C * np.clip(1 - (r / R) ** 2, 0, None))

    def hr(self, r, t):
        """dh/dr (analytic; -inf at the front)"""
        R = self.R(t)
        s = np.clip(1 - (r / R) ** 2, 1e-12, None)
        return self.A(t) * C ** (1 / 3) * (1 / 3) * s ** (-2 / 3) * (-2 * r / R ** 2)

    def Q(self, r, z, t):
        return 3 * self.D * (-self.hr(r, t)) * (self.h(r, t) * z ** 2 / 2 - z ** 3 / 6)

    def vel(self, r, z, t):
        h = self.h(r, t)
        z = np.clip(z, 0, h)
        u = 3 * self.D * (-self.hr(r, t)) * (h * z - z ** 2 / 2)
        dr = 1e-4 * self.R(t)
        rp, rm = r + dr, np.maximum(r - dr, 0.0)
        w = -(rp * self.Q(rp, np.minimum(z, self.h(rp, t)), t) - rm * self.Q(rm, np.minimum(z, self.h(rm, t)), t)) \
            / ((rp - rm) * np.maximum(r, dr))
        return u, w


def main():
    args = sys.argv[1:]
    cfgf = arg(args, "--config", os.path.join(ROOT, "configs", "spread_drop_f0500.toml"))
    seed = arg(args, "--seed", 200.0, float)
    tmax = arg(args, "--tmax", 1e6, float)
    n = arg(args, "--n", 60, int)
    depths = [float(v) for v in arg(args, "--depths", "0,10,20").split(",")]
    nfr = arg(args, "--frames", 40, int)
    out = arg(args, "--out", os.path.join(os.getcwd(), "huppert_tracers.html"))
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    c = tomllib.load(open(cfgf, "rb"))["case"]
    P = eval_lua_parameters(os.path.join(ROOT, c.get("parameters", "case/parameters.lua")))
    D = abs(P["rhoi"] * P["gravity"]) / (3 * P["viscosity"])
    A0, R0 = float(c["drop_height_m"]), float(c["drop_radius_km"]) * 1e3
    V = (0.75 if str(c.get("drop_shape", "smooth")) == "huppert" else 1 / 3) * np.pi * R0 ** 2 * A0
    hp = Huppert(V, D)
    tv = (R0 / XI) ** 8 / (D * V ** 3)
    print(f"Huppert drop: V {V:.3g} m^3, D {D:.3g} /(m yr), virtual origin {tv:,.0f} yr; R {hp.R(tv + seed) / 1e3:.2f} km "
          f"at t = {seed:g} yr, {hp.R(tv + tmax) / 1e3:.2f} km at {tmax:g} yr")
    # seeds: n along the surface (r up to 0.98 R), at each depth
    Rs = hp.R(tv + seed)
    r0 = np.linspace(0.0, 0.98 * Rs, n)
    hs = hp.h(r0, tv + seed)
    rr = np.concatenate([r0 for _ in depths])
    zz = np.concatenate([np.maximum(hs - d, 0.0) for d in depths])
    from scipy.integrate import solve_ivp
    m = len(rr)

    def rhs(t, y):
        r, z = y[:m], y[m:]
        u, w = hp.vel(np.clip(r, 0, None), z, t + tv)
        R = hp.R(t + tv)
        stuck = r >= R                       # at (beyond) the nose: on the bed, overridden: at rest
        u = np.where(stuck, 0.0, u)
        w = np.where(stuck, 0.0, w)
        return np.r_[u, w]

    times = np.unique(np.r_[seed, np.geomspace(max(seed, 1.0), tmax, nfr)])
    sol = solve_ivp(rhs, (seed, tmax), np.r_[rr, zz], t_eval=times, rtol=1e-6, atol=1e-3, method="RK45")
    Rt = np.array([hp.R(t + tv) for t in sol.t])
    Y = sol.y
    rs, zs = Y[:m], np.maximum(Y[m:], 0.0)
    # clamp into the drop (the surface is material; the numerics drift a little)
    for k, t in enumerate(sol.t):
        zs[:, k] = np.minimum(zs[:, k], hp.h(np.minimum(rs[:, k], Rt[k]), t + tv))
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    col = np.tile(np.arange(n), len(depths))
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, row_heights=[0.55, 0.45],
                        subplot_titles=("the drop and its tracers (slider: time)",
                                        "the tracers' paths in the similarity coordinates (r/R, z/h)"))
    nt = len(sol.t)
    rmax = 1.05 * Rt[-1]
    for k, t in enumerate(sol.t):
        rg = np.linspace(0, Rt[k], 300)
        fig.add_trace(go.Scatter(x=rg / 1e3, y=hp.h(rg, t + tv), mode="lines", line=dict(color="black"),
                                 visible=k == 0, showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=rs[:, k] / 1e3, y=zs[:, k], mode="markers",
                                 marker=dict(size=5, color=col, colorscale="Turbo", line=dict(width=0.4, color="black")),
                                 visible=k == 0, showlegend=False,
                                 hovertemplate="tracer %{marker.color}<br>r %{x:.2f} km, z %{y:.1f} m<extra></extra>"),
                      row=1, col=1)
    # the similarity-space paths (static)
    for j in range(m):
        xi = rs[j] / Rt
        h_ = np.array([hp.h(min(rs[j, k], Rt[k]), sol.t[k] + tv) for k in range(nt)])
        ze = np.where(h_ > 0, zs[j] / np.maximum(h_, 1e-9), 0.0)
        ze = np.where(xi >= 1, 0.0, ze)
        fig.add_trace(go.Scatter(x=xi, y=ze, mode="lines+markers", marker=dict(size=3),
                                 line=dict(width=1, color=f"hsl({int(300 * (j % n) / max(n - 1, 1))},80%,45%)"),
                                 showlegend=False, hovertemplate=f"tracer {j % n}, depth {depths[j // n]:g} m"
                                                                 "<br>r/R %{x:.3f}, z/h %{y:.3f}<extra></extra>"),
                      row=2, col=1)
    steps = [dict(label=f"{t:,.0f}", method="restyle",
                  args=[{"visible": [i // 2 == k for i in range(2 * nt)]}, list(range(2 * nt))]) for k, t in
             enumerate(sol.t)]
    fig.update_xaxes(title_text="r (km)", range=[0, rmax / 1e3], row=1, col=1)
    fig.update_yaxes(title_text="z (m)", range=[0, 1.05 * hp.h(0, tv + seed)], row=1, col=1)
    fig.update_xaxes(title_text="r / R", range=[0, 1.02], row=2, col=1)
    fig.update_yaxes(title_text="z / h", range=[0, 1.02], row=2, col=1)
    fig.update_layout(template="plotly_white", height=1000,
                      title=f"Huppert's drop (dry no-slip bed; V {V:.3g} m^3, eta {P['viscosity'] * 1e6 * 31557600:.2g} "
                            f"Pa s): tracers from the surface at {seed:g} yr to {tmax:g} yr",
                      sliders=[dict(active=0, currentvalue=dict(prefix="t (yr) = "), steps=steps, pad=dict(t=40))])
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out}")
    # a summary: the fraction of tracers rolled under (at the nose) by each time
    for k in np.linspace(0, nt - 1, 6).astype(int):
        print(f"t {sol.t[k]:>10,.0f} yr: R {Rt[k] / 1e3:6.2f} km, tracers at the nose (rolled under) "
              f"{np.mean(rs[:, k] >= 0.999 * Rt[k]):.0%}")


if __name__ == "__main__":
    main()
