#!/usr/bin/env python3
"""
Strain rates at the base, map view (plotly), from a frames file (plots/vtu_frames.py): which terms dominate where,
and what each approximation (SIA, SSA, Blatter-Pattyn, full Stokes) keeps and drops.

    python3 plots/base_strain.py FRAMES.npz [--frame K | --t YEARS] [--res KM] [--x XMIN XMAX] [--yscale S]
        [--shear-factor F] [--out FILE.html]

From the base velocity (u, v, w) the horizontal derivatives along the base: du/dx, du/dy, dv/dx, dv/dy, dw/dx, dw/dy
(gradients on the triangulated base nodes, averaged to the nodes). The frames hold no velocity inside the ice, so the
vertical shear du/dz, dv/dz is only estimated, as the column mean (u_s - u_b) / H from the surface and base velocity
(times --shear-factor F: 1 = the column mean; for shallow-ice flow with n = 3 the shear at the bed is (n+1) = 4 times
the mean). dw/dz = -(du/dx + dv/dy) (incompressible).

Strain-rate components (1/yr):
    e_xx = du/dx    e_yy = dv/dy    e_zz = dw/dz = -(e_xx + e_yy)    e_xy = (du/dy + dv/dx) / 2
    e_xz = (du/dz + dw/dx) / 2      e_yz = (dv/dz + dw/dy) / 2
What the approximations keep:
    SIA (shallow ice):       du/dz, dv/dz only (vertical shear); drops e_xx, e_yy, e_xy and dw/dx, dw/dy
    SSA (shallow shelf):     e_xx, e_yy, e_xy (membrane stresses, depth-uniform); drops the vertical shear and dw/dx, dw/dy
    hybrid / L1L2:           SSA + vertical shear; drops dw/dx, dw/dy
    Blatter-Pattyn (HO/FO):  everything except dw/dx, dw/dy in e_xz, e_yz (and the pressure is hydrostatic)
    full Stokes:             everything
The maps (the dropdown): each component; the derivatives dw/dx, dw/dy alone; "membrane share" = |membrane| /
(|membrane| + |shear|) (0: SIA-like, 1: SSA-like; |membrane| = sqrt(e_xx^2 + e_yy^2 + e_xx e_yy + e_xy^2), |shear| =
sqrt(e_xz^2 + e_yz^2)); "dw/dx vs du/dz" = |grad w| / |du/dz, dv/dz| (large: Blatter-Pattyn's approximation fails);
and the dominant term at each point. The grounding line in black.

Note: the derivatives are along the base, not at constant z. Where the bed slopes, d/dx along the base = d/dx at fixed
z + slope x d/dz, so on steep beds the along-base du/dx includes some vertical shear.
"""
import os
import sys

import numpy as np


def opt(args, name, n=1, conv=float, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = [conv(a) for a in args[k + 1:k + 1 + n]]
    return args[:k] + args[k + 1 + n:], (v[0] if n == 1 else v)


def node_gradients(tri, x, y, fields):
    """d/dx, d/dy of each nodal field: per triangle (linear), then the area-weighted mean over the triangles at a node"""
    T = tri.simplices
    x1, y1 = x[T[:, 0]], y[T[:, 0]]
    dx21, dy21 = x[T[:, 1]] - x1, y[T[:, 1]] - y1
    dx31, dy31 = x[T[:, 2]] - x1, y[T[:, 2]] - y1
    det = dx21 * dy31 - dx31 * dy21
    area = 0.5 * np.abs(det)
    good = area > 1e-6 * np.median(area)
    det = np.where(good, det, np.nan)
    wsum = np.bincount(T[good].ravel(), weights=np.repeat(area[good], 3), minlength=len(x))
    out = {}
    for name, f in fields.items():
        f1 = f[T[:, 0]]
        df21, df31 = f[T[:, 1]] - f1, f[T[:, 2]] - f1
        gx = (df21 * dy31 - df31 * dy21) / det
        gy = (df31 * dx21 - df21 * dx31) / det
        nx = np.bincount(T[good].ravel(), weights=np.repeat((gx * area)[good], 3), minlength=len(x))
        ny = np.bincount(T[good].ravel(), weights=np.repeat((gy * area)[good], 3), minlength=len(x))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = (nx / wsum, ny / wsum)
    return out


def main():
    args = sys.argv[1:]
    args, frame = opt(args, "--frame", conv=int)
    args, tsel = opt(args, "--t")
    args, res = opt(args, "--res", default=1.0)
    args, xwin = opt(args, "--x", n=2)
    args, yscale = opt(args, "--yscale", default=3.0)
    args, sf = opt(args, "--shear-factor", default=1.0)
    args, out = opt(args, "--out", conv=str)
    if len(args) != 1:
        raise SystemExit(__doc__)
    path = args[0]
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator
    import plotly.graph_objects as go

    d = np.load(path, allow_pickle=True)
    times = d["frame_times"]
    k = len(times) - 1 if frame is None else frame
    if tsel is not None:
        k = int(np.argmin(np.abs(times - tsel)))
    t = float(times[k])
    x, y = d["base_x"].astype(float), d["base_y"].astype(float)
    u, v, w = (d[f"base_v{c}"][k].astype(float) for c in "xyz")
    mask = d["base_mask"][k].astype(float)
    H = None
    tx, ty = d["top_x"].astype(float), d["top_y"].astype(float)
    if len(tx) == len(x) and np.allclose(tx, x) and np.allclose(ty, y):
        us, vs = d["top_vx"][k].astype(float), d["top_vy"][k].astype(float)
        H = d["top_z"][k].astype(float) - d["base_z"][k].astype(float)
    else:                                          # different node sets: the surface interpolated to the base nodes
        tt = Delaunay(np.c_[tx, ty])
        us = LinearNDInterpolator(tt, d["top_vx"][k].astype(float))(x, y)
        vs = LinearNDInterpolator(tt, d["top_vy"][k].astype(float))(x, y)
        H = LinearNDInterpolator(tt, d["top_z"][k].astype(float))(x, y) - d["base_z"][k].astype(float)

    tri = Delaunay(np.c_[x, y])
    g = node_gradients(tri, x, y, {"u": u, "v": v, "w": w})
    dudx, dudy = g["u"]; dvdx, dvdy = g["v"]; dwdx, dwdy = g["w"]
    with np.errstate(invalid="ignore", divide="ignore"):
        dudz = sf * (us - u) / np.where(H > 1.0, H, np.nan)
        dvdz = sf * (vs - v) / np.where(H > 1.0, H, np.nan)
    exx, eyy = dudx, dvdy
    ezz = -(exx + eyy)
    exy = 0.5 * (dudy + dvdx)
    exz = 0.5 * (dudz + dwdx)
    eyz = 0.5 * (dvdz + dwdy)
    memb = np.sqrt(np.maximum(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2, 0))
    shear = np.sqrt(exz ** 2 + eyz ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mshare = memb / (memb + shear)
        sh = np.hypot(dudz, dvdz)
        wratio = np.where(sh > 0, np.hypot(dwdx, dwdy) / sh, np.nan)     # no vertical shear (plug flow): undefined
    # the dominant term: the largest of the terms an approximation would keep or drop
    terms = [("e_xx", exx), ("e_yy", eyy), ("e_xy", exy), ("du/dz, dv/dz / 2", 0.5 * np.hypot(dudz, dvdz)),
             ("dw/dx, dw/dy / 2", 0.5 * np.hypot(dwdx, dwdy))]
    A = np.abs(np.c_[tuple(a for _, a in terms)])
    dom = np.where(np.isfinite(A).all(axis=1), np.nanargmax(np.nan_to_num(A, nan=-1), axis=1), -1).astype(float)
    dom[dom < 0] = np.nan

    # onto a regular grid (km)
    x0, x1 = (xwin if xwin else (x.min() / 1e3, x.max() / 1e3))
    gx = np.arange(x0, x1 + 1e-9, res)
    gy = np.arange(y.min() / 1e3, y.max() / 1e3 + 1e-9, min(res, (y.max() - y.min()) / 1e3 / 20))
    GX, GY = np.meshgrid(gx, gy)
    pts = np.c_[GX.ravel() * 1e3, GY.ravel() * 1e3]

    def grid(f):
        return LinearNDInterpolator(tri, f)(pts).reshape(GX.shape).astype(np.float32)

    def gnear(f):                                  # categorical: the nearest node, not interpolated
        from scipy.spatial import cKDTree
        _, i = cKDTree(np.c_[x, y]).query(pts)
        return f[i].reshape(GX.shape).astype(np.float32)

    signed = [("e_xx = du/dx", exx), ("e_yy = dv/dy", eyy), ("e_zz = dw/dz = -(e_xx + e_yy)", ezz),
              ("e_xy = (du/dy + dv/dx)/2", exy), ("e_xz = (du/dz + dw/dx)/2", exz), ("e_yz = (dv/dz + dw/dy)/2", eyz),
              ("du/dz (est.)", dudz), ("dv/dz (est.)", dvdz), ("dw/dx", dwdx), ("dw/dy", dwdy),
              ("du/dy", dudy), ("dv/dx", dvdx)]
    traces, labels = [], []
    for name, f in signed:
        G = grid(f)
        a = np.nanpercentile(np.abs(G), 99) if np.isfinite(G).any() else 1.0
        traces.append(go.Heatmap(x=gx, y=gy, z=G, colorscale="RdBu_r", zmid=0, zmin=-a, zmax=a,
                                 colorbar=dict(title="1/yr"), hovertemplate=f"{name}: %{{z:.3g}} /yr<br>x %{{x:.1f}} "
                                 "y %{y:.1f} km<extra></extra>"))
        labels.append(name)
    for name, f in [("|membrane| (SSA terms)", memb), ("|shear| (e_xz, e_yz)", shear)]:
        G = grid(np.log10(np.maximum(f, 1e-12)))
        lo, hi = np.nanpercentile(G, [1, 99.5]) if np.isfinite(G).any() else (-6, 0)
        traces.append(go.Heatmap(x=gx, y=gy, z=G, colorscale="Viridis", zmin=lo, zmax=hi,
                                 colorbar=dict(title="log10 1/yr"),
                                 hovertemplate=f"log10 {name}: %{{z:.2f}}<br>x %{{x:.1f}} y %{{y:.1f}} km<extra></extra>"))
        labels.append(f"log10 {name}")
    traces.append(go.Heatmap(x=gx, y=gy, z=grid(mshare), colorscale="Cividis", zmin=0, zmax=1,
                             colorbar=dict(title="0 SIA .. 1 SSA"),
                             hovertemplate="membrane share %{z:.2f}<br>x %{x:.1f} y %{y:.1f} km<extra></extra>"))
    labels.append("membrane share |memb| / (|memb| + |shear|)")
    G = grid(np.log10(np.maximum(wratio, 1e-6)))
    traces.append(go.Heatmap(x=gx, y=gy, z=G, colorscale="RdBu_r", zmid=0, zmin=-3, zmax=1,
                             colorbar=dict(title="log10 ratio"),
                             hovertemplate="log10 |grad w| / |du/dz|: %{z:.2f}<br>x %{x:.1f} y %{y:.1f} km<extra></extra>"))
    labels.append("log10 |dw/dx, dw/dy| / |du/dz, dv/dz| (Blatter-Pattyn drops the numerator)")
    cols = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e"]
    nT = len(terms)
    cs = []
    for i, c in enumerate(cols[:nT]):
        cs += [[i / nT, c], [(i + 1) / nT, c]]
    traces.append(go.Heatmap(x=gx, y=gy, z=gnear(dom), colorscale=cs, zmin=-0.5, zmax=nT - 0.5,
                             colorbar=dict(title="dominant", tickvals=list(range(nT)), ticktext=[n for n, _ in terms]),
                             hovertemplate="dominant: %{z}<br>x %{x:.1f} y %{y:.1f} km<extra></extra>"))
    labels.append("dominant term")

    for i, tr in enumerate(traces):
        tr.visible = (i == 0)
    gl = go.Contour(x=gx, y=gy, z=grid(mask), contours=dict(start=-0.5, end=-0.5, size=1, coloring="none"),
                    line=dict(color="black", width=2), showscale=False, hoverinfo="skip", name="grounding line")
    fig = go.Figure(traces + [gl])
    n = len(traces)
    buttons = [dict(label=lab, method="update", args=[{"visible": [j == i for j in range(n)] + [True]},
                                                     {"title.text": f"{os.path.basename(path)}, t = {t:,.0f} yr: {lab}"}])
               for i, lab in enumerate(labels)]
    fig.update_layout(
        title=f"{os.path.basename(path)}, t = {t:,.0f} yr: {labels[0]}",
        updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.12, xanchor="left", yanchor="bottom")],
        xaxis=dict(title="x (km)"), yaxis=dict(title=f"y (km, x{yscale:g})", scaleanchor="x", scaleratio=yscale),
        template="plotly_white", margin=dict(t=110),
        annotations=[dict(x=1, y=1.12, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                          align="right", font=dict(size=11),
                          text="SIA: du/dz, dv/dz only | SSA: e_xx, e_yy, e_xy | Blatter-Pattyn: all but dw/dx, dw/dy | "
                               f"du/dz = {sf:g} x (u_s - u_b)/H")])
    out = out or os.path.splitext(path)[0] + f"_strain_t{t:.0f}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"frame {k} (t {t:g} yr) -> {out}")
    try:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out))
    except Exception:
        pass


if __name__ == "__main__":
    main()
