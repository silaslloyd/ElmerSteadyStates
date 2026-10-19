#!/usr/bin/env python3
"""
Look at a steady state or an eigenmode of the Elmer flowline:
  * basal temperature and surface elevation along x,
  * the temperature field (or the mode's temperature component) on the (x, z) mesh nodes,
  * the along-flow and vertical velocities u, w on the mesh nodes, if the file has them (states from
    `ess velocity`, <state>_vel.npz; for a plain state the _vel file next to it is used when present).

FILE is a driver state (.npz from newton / continue / extract) or a stability.npz (then MODE picks
the eigenvector; its real part is shown, normalised to max |T'| = 1).

Press Run in an IDE (settings below), or:  python plots/plot_state.py work/land_flowline/steady_p266.npz
Needs numpy and plotly. Linear axes.
"""
import glob
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ess.statefile import StateLayout, find_name  # noqa: E402

# ---- settings for pressing Run
FILE = None          # None -> the newest .npz under work/ (state or stability)
MODE = 0             # eigenvector index for stability files
Y_SIDE = "min"       # the flowline mesh is one element wide: show the nodes at y = min (or "max")
SHOW = True


def load(path, mode):
    z = np.load(path, allow_pickle=False)
    layout = StateLayout.from_npz(z)
    diag = {k.split("::", 1)[1]: (z[k], z["diag_vals::" + k.split("::", 1)[1]])
            for k in z.files if k.startswith("diag_gids::")}
    frozen = {k.split("::", 1)[1]: (z[k], z["frozen_vals::" + k.split("::", 1)[1]])
              for k in z.files if k.startswith("frozen_gids::")}
    if "vectors" in z.files:
        v = z["vectors"][:, mode]
        mu = z["mu"][mode]
        sig = z["sigma"][mode]
        T = layout.split(v.real)
        tmax = np.max(np.abs(T[layout.names[0]]))
        x = {k: a / tmax for k, a in T.items()}
        title = f"mode {mode}: mu = {mu:.6f}, sigma = {sig.real:+.4e} {sig.imag:+.3e}i (real part)"
        return layout, x, diag, frozen, title, True
    return layout, layout.split(z["x"]), diag, frozen, f"state at p = {float(z['p']):.6g}", False


def on_side(gids, coords):
    gy, y = coords["coordinate 2"]
    k = np.searchsorted(gy, gids)
    yv = y[np.clip(k, 0, len(y) - 1)]
    target = yv.min() if Y_SIDE == "min" else yv.max()
    return np.isclose(yv, target)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else FILE
    if path is None:
        files = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "work", "**", "*.npz"), recursive=True),
                       key=os.path.getmtime)
        files = [f for f in files if "cache" not in f and "arnoldi" not in f and "/pt_" not in f
                 and "checkpoint" not in f]
        if not files:
            raise SystemExit("no state files under work/")
        path = files[-1]
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    layout, fields, diag, frozen, title, is_mode = load(path, MODE)
    vel = {}
    vpath = os.path.splitext(path)[0] + "_vel.npz"
    vdiag = load(vpath, 0)[2] if not is_mode and not path.endswith("_vel.npz") and os.path.exists(vpath) else diag
    for c in (1, 3):
        try:
            vel[c] = vdiag[find_name(vdiag, f"velocity {c}")]
        except KeyError:
            pass
    vrows = not is_mode and len(vel) == 2
    coords = {f"coordinate {c}": diag[find_name(diag, f"coordinate {c}")] for c in (1, 2, 3)}
    gx, X = coords["coordinate 1"]
    gz, Z = coords["coordinate 3"]
    tname = next(n for n in layout.names if n.lower() == "temperature")
    zname = next((n for n in layout.names if n.lower() == "zs"), None)
    gT = next(b.gids for b in layout.blocks if b.name == tname)

    def xy(gids):
        return X[np.searchsorted(gx, gids)], Z[np.searchsorted(gz, gids)]

    titles = ["basal temperature" + (" (mode)" if is_mode else ", K"),
              "surface elevation" + (" (mode)" if is_mode else ", m"), "temperature on the mesh nodes"]
    heights = [0.25, 0.25, 0.5]
    if vrows:
        titles += ["along-flow velocity u (m/yr)", "vertical velocity w (m/yr)"]
        heights = [0.15, 0.15, 0.24, 0.23, 0.23]
    rows = len(titles)
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.05, row_heights=heights, subplot_titles=titles)
    # basal nodes = the nodes carrying the frozen masks (or Zb)
    src = frozen or diag
    bkey = next((k for k in src if k.lower() in ("coldtempmask", "zb")), None)
    if bkey is not None:
        gb = src[bkey][0]
        gb = gb[on_side(gb, coords)]
        j = np.searchsorted(gT, gb)
        xb = xy(gb)[0]
        o = np.argsort(xb)
        fig.add_trace(go.Scatter(x=xb[o], y=fields[tname][j][o], mode="lines", name="T bed"), 1, 1)
        if not is_mode:
            fig.add_hline(y=273.15, line=dict(dash="dot", color="grey"), row=1, col=1)
    if zname:
        gs = next(b.gids for b in layout.blocks if b.name == zname)
        sel = on_side(gs, coords)
        xs = xy(gs[sel])[0]
        o = np.argsort(xs)
        fig.add_trace(go.Scatter(x=xs[o], y=fields[zname][sel][o], mode="lines", name="Zs"), 2, 1)
    sel = on_side(gT, coords)
    xt, zt = xy(gT[sel])
    fig.add_trace(go.Scattergl(x=xt, y=zt, mode="markers",
                               marker=dict(color=fields[tname][sel], size=4, colorscale="RdBu_r" if is_mode else "Viridis",
                                           colorbar=dict(title="T", y=0.58 if vrows else 0.25, len=0.22 if vrows else 0.5),
                                           cmid=0 if is_mode else None),
                               hovertemplate="x %{x:.4g}<br>z %{y:.4g}<br>T %{marker.color:.5g}<extra></extra>",
                               name="T"), 3, 1)
    if vrows:
        bar_y = [None, None, None, 0.35, 0.11]
        for r, c, nm, cs in ((4, 1, "u", "Viridis"), (5, 3, "w", "RdBu_r")):
            gv, vv = vel[c]
            sv = on_side(gv, coords)
            xv, zv = xy(gv[sv])
            val = vv[sv]
            lim = np.abs(val).max() or 1.0
            fig.add_trace(go.Scattergl(
                x=xv, y=zv, mode="markers", name=nm,
                marker=dict(color=val, size=4, colorscale=cs, cmin=-lim if nm == "w" else None,
                            cmax=lim if nm == "w" else None,
                            colorbar=dict(title=f"{nm} (m/yr)", y=bar_y[r - 1], len=0.2)),
                hovertemplate=f"x %{{x:.4g}}<br>z %{{y:.4g}}<br>{nm} %{{marker.color:.4g}} m/yr<extra></extra>"), r, 1)
    fig.update_xaxes(title_text="x (m)", row=rows, col=1)
    fig.update_layout(title=f"{os.path.basename(path)}: {title}", height=1000 if rows == 3 else 1600, template="plotly_white",
                      showlegend=False)
    out = os.path.splitext(path)[0] + (f"_mode{MODE}" if is_mode else "") + ".html"
    fig.write_html(out, include_plotlyjs="cdn")
    print("wrote", out)
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
