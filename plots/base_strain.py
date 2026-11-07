#!/usr/bin/env python3
"""
Strain rates at the base, map view (plotly), from a frames file (plots/vtu_frames.py): which terms dominate where,
and what each approximation (SIA, SSA, Blatter-Pattyn, full Stokes) keeps and drops.

    python3 plots/base_strain.py FRAMES.npz [--every N | --frame K | --t YEARS] [--res KM] [--x XMIN XMAX]
        [--yscale S] [--shear-factor F] [--out FILE.html]

Default: every 5th frame (--every N; the last always), a slider through them and a dropdown for the map; --frame /
--t: one frame. Grounded ice only (the grounding line and the shelf masked out, gradients from grounded triangles);
--keep-shelf: everything. --res: the map grid in km (default 2 with a slider, 1 for one frame); --x: only that x range.

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
import json
import os
import sys

import numpy as np


def opt(args, name, n=1, conv=float, default=None):
    if name not in args:
        return args, default
    k = args.index(name)
    v = [conv(a) for a in args[k + 1:k + 1 + n]]
    return args[:k] + args[k + 1 + n:], (v[0] if n == 1 else v)


def node_gradients(tri, x, y, fields, use=None):
    """d/dx, d/dy of each nodal field: per triangle (linear), then the area-weighted mean over the triangles at a node"""
    T = tri.simplices
    x1, y1 = x[T[:, 0]], y[T[:, 0]]
    dx21, dy21 = x[T[:, 1]] - x1, y[T[:, 1]] - y1
    dx31, dy31 = x[T[:, 2]] - x1, y[T[:, 2]] - y1
    det = dx21 * dy31 - dx31 * dy21
    area = 0.5 * np.abs(det)
    good = area > 1e-6 * np.median(area)
    if use is not None:                            # only the triangles with all their nodes in use (e.g. grounded)
        good &= use[T].all(axis=1)
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


def frame_fields(d, k, x, y, tri, sf, interp_top, grounded_only=True):
    """the maps of frame k at the base nodes: [(label, values, kind)], kind: signed | log | share | ratio | cat"""
    u, v, w = (d[f"base_v{c}"][k].astype(float) for c in "xyz")
    us, vs, H = interp_top(k)
    # grounded only: the grounding line (GroundedMask 0) and the shelf (-1) masked out, and the gradients from the
    # grounded triangles only (not across the jump at the grounding line)
    gm = d["base_mask"][k].astype(float)
    use = gm > 0.5 if grounded_only else None
    g = node_gradients(tri, x, y, {"u": u, "v": v, "w": w}, use)
    dudx, dudy = g["u"]; dvdx, dvdy = g["v"]; dwdx, dwdy = g["w"]
    with np.errstate(invalid="ignore", divide="ignore"):
        Hs = np.where(H > 1.0, H, np.nan)
        dudz, dvdz = sf * (us - u) / Hs, sf * (vs - v) / Hs
    exx, eyy = dudx, dvdy
    exy = 0.5 * (dudy + dvdx)
    exz, eyz = 0.5 * (dudz + dwdx), 0.5 * (dvdz + dwdy)
    memb = np.sqrt(np.maximum(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2, 0))
    shear = np.sqrt(exz ** 2 + eyz ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mshare = memb / (memb + shear)
        sh = np.hypot(dudz, dvdz)
        wratio = np.where(sh > 0, np.hypot(dwdx, dwdy) / sh, np.nan)     # no vertical shear (plug flow): undefined
    A = np.abs(np.c_[tuple(a for _, a in dominant_terms(exx, eyy, exy, dudz, dvdz, dwdx, dwdy))])
    dom = np.where(np.isfinite(A).all(axis=1), np.argmax(np.nan_to_num(A, nan=-1), axis=1), -1).astype(float)
    dom[dom < 0] = np.nan
    if grounded_only:
        off = ~use
        for a in (exx, eyy, exy, exz, eyz, dudz, dvdz, dwdx, dwdy, dudy, dvdx, memb, shear, mshare, wratio, dom):
            a[off] = np.nan
    return [("e_xx = du/dx", exx, "signed"), ("e_yy = dv/dy", eyy, "signed"),
            ("e_zz = dw/dz = -(e_xx + e_yy)", -(exx + eyy), "signed"), ("e_xy = (du/dy + dv/dx)/2", exy, "signed"),
            ("e_xz = (du/dz + dw/dx)/2", exz, "signed"), ("e_yz = (dv/dz + dw/dy)/2", eyz, "signed"),
            ("du/dz (est.)", dudz, "signed"), ("dv/dz (est.)", dvdz, "signed"), ("dw/dx", dwdx, "signed"),
            ("dw/dy", dwdy, "signed"), ("du/dy", dudy, "signed"), ("dv/dx", dvdx, "signed"),
            ("log10 |membrane| (SSA terms)", np.log10(np.maximum(memb, 1e-12)), "log"),
            ("log10 |shear| (e_xz, e_yz)", np.log10(np.maximum(shear, 1e-12)), "log"),
            ("membrane share |memb| / (|memb| + |shear|)", mshare, "share"),
            ("log10 |dw/dx, dw/dy| / |du/dz, dv/dz| (Blatter-Pattyn drops the numerator)",
             np.log10(np.maximum(wratio, 1e-6)), "ratio"),
            ("dominant term", dom, "cat")]


def dominant_terms(exx, eyy, exy, dudz, dvdz, dwdx, dwdy):
    return [("e_xx", exx), ("e_yy", eyy), ("e_xy", exy), ("du/dz, dv/dz / 2", 0.5 * np.hypot(dudz, dvdz)),
            ("dw/dx, dw/dy / 2", 0.5 * np.hypot(dwdx, dwdy))]


def main():
    args = sys.argv[1:]
    args, frame = opt(args, "--frame", conv=int)
    args, tsel = opt(args, "--t")
    args, every = opt(args, "--every", conv=int, default=5)
    args, res = opt(args, "--res")
    args, xwin = opt(args, "--x", n=2)
    args, yscale = opt(args, "--yscale", default=3.0)
    args, sf = opt(args, "--shear-factor", default=1.0)
    args, out = opt(args, "--out", conv=str)
    grounded_only = "--keep-shelf" not in args            # --keep-shelf: the grounding line and the shelf too
    args = [a for a in args if a != "--keep-shelf"]
    if len(args) != 1:
        raise SystemExit(__doc__)
    path = args[0]
    from scipy.spatial import Delaunay, cKDTree
    from scipy.interpolate import LinearNDInterpolator
    import plotly.graph_objects as go

    d = np.load(path, allow_pickle=True)
    times = d["frame_times"]
    if frame is not None or tsel is not None:      # one frame
        ks = [int(np.argmin(np.abs(times - tsel)))] if tsel is not None else [frame % len(times)]
    else:                                          # every N-th frame, and the last
        ks = sorted(set(list(range(0, len(times), every)) + [len(times) - 1]))
    res = res or (1.0 if len(ks) == 1 else 2.0)    # km; coarser with a slider (the page holds every map of every frame)
    x, y = d["base_x"].astype(float), d["base_y"].astype(float)
    tx, ty = d["top_x"].astype(float), d["top_y"].astype(float)
    same = len(tx) == len(x) and np.allclose(tx, x) and np.allclose(ty, y)
    tt = None if same else Delaunay(np.c_[tx, ty])

    def interp_top(k):                             # the surface velocity and the thickness at the base nodes
        if same:
            return (d["top_vx"][k].astype(float), d["top_vy"][k].astype(float),
                    d["top_z"][k].astype(float) - d["base_z"][k].astype(float))
        f = lambda a: LinearNDInterpolator(tt, a.astype(float))(x, y)
        return f(d["top_vx"][k]), f(d["top_vy"][k]), f(d["top_z"][k]) - d["base_z"][k].astype(float)

    tri = Delaunay(np.c_[x, y])
    x0, x1 = (xwin if xwin else (x.min() / 1e3, x.max() / 1e3))
    gx = np.arange(x0, x1 + 1e-9, res)
    gy = np.arange(y.min() / 1e3, y.max() / 1e3 + 1e-9, min(res, (y.max() - y.min()) / 1e3 / 20))
    GX, GY = np.meshgrid(gx, gy)
    pts = np.c_[GX.ravel() * 1e3, GY.ravel() * 1e3]
    _, near = cKDTree(np.c_[x, y]).query(pts)

    def grid(f, cat=False):
        if cat:                                    # categorical: the nearest node, not interpolated
            return f[near].reshape(GX.shape).astype(np.float32)
        return LinearNDInterpolator(tri, f)(pts).reshape(GX.shape).astype(np.float32)

    maps, glz = [], []                             # maps[frame][field] = (label, grid, kind)
    for n, k in enumerate(ks):
        maps.append([(lab, grid(f, kind == "cat"), kind) for lab, f, kind in frame_fields(d, k, x, y, tri, sf, interp_top,
                                                                                   grounded_only)])
        glz.append(grid(d["base_mask"][k].astype(float)))
        print(f"  frame {k} (t {float(times[k]):g} yr) {n + 1}/{len(ks)}", end="\r", flush=True)
    print()
    nF = len(maps[0])
    labels = [m[0] for m in maps[0]]
    # one colour range per field, over all the frames shown
    rng = []
    for j in range(nF):
        kind = maps[0][j][2]
        allz = np.concatenate([m[j][1].ravel() for m in maps])
        allz = allz[np.isfinite(allz)]
        if kind == "signed":
            a = float(np.percentile(np.abs(allz), 99)) if allz.size else 1.0
            rng.append(dict(colorscale="RdBu_r", zmid=0, zmin=-a, zmax=a, colorbar=dict(title="1/yr")))
        elif kind == "log":
            lo, hi = np.percentile(allz, [1, 99.5]) if allz.size else (-6, 0)
            rng.append(dict(colorscale="Viridis", zmin=float(lo), zmax=float(hi), colorbar=dict(title="log10 1/yr")))
        elif kind == "share":
            rng.append(dict(colorscale="Cividis", zmin=0, zmax=1, colorbar=dict(title="0 SIA .. 1 SSA")))
        elif kind == "ratio":
            rng.append(dict(colorscale="RdBu_r", zmid=0, zmin=-3, zmax=1, colorbar=dict(title="log10 ratio")))
        else:
            names = [n_ for n_, _ in dominant_terms(*([np.zeros(1)] * 7))]
            cols = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e"]
            cs = []
            for i, c in enumerate(cols[:len(names)]):
                cs += [[i / len(names), c], [(i + 1) / len(names), c]]
            rng.append(dict(colorscale=cs, zmin=-0.5, zmax=len(names) - 0.5,
                            colorbar=dict(title="dominant", tickvals=list(range(len(names))), ticktext=names)))

    traces = []
    for n, m in enumerate(maps):                   # frame-major: trace n * nF + j
        for j, (lab, Z, kind) in enumerate(m):
            traces.append(go.Heatmap(x=gx, y=gy, z=Z, visible=(n == len(maps) - 1 and j == 0), **rng[j],
                                     hovertemplate=f"{lab}: %{{z:.3g}}<br>x %{{x:.1f}} y %{{y:.1f}} km<extra></extra>"))
    for n, Z in enumerate(glz):                    # then the grounding line of each frame
        traces.append(go.Contour(x=gx, y=gy, z=Z, contours=dict(start=-0.5, end=-0.5, size=1, coloring="none"),
                                 line=dict(color="black", width=2), showscale=False, hoverinfo="skip",
                                 visible=(n == len(maps) - 1), name="grounding line"))
    tlab = [f"t {float(times[k]):,.0f} yr" for k in ks]
    fig = go.Figure(traces)
    layout = dict(
        title=f"{os.path.basename(path)}, {tlab[-1]}: {labels[0]}",
        updatemenus=[dict(buttons=[dict(label=lab, method="skip", args=[None]) for lab in labels], direction="down",
                          x=0, y=1.12, xanchor="left", yanchor="bottom", active=0)],
        xaxis=dict(title="x (km)"), yaxis=dict(title=f"y (km, x{yscale:g})", scaleanchor="x", scaleratio=yscale),
        template="plotly_white", margin=dict(t=110),
        annotations=[dict(x=1, y=1.12, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                          align="right", font=dict(size=11),
                          text="SIA: du/dz, dv/dz only | SSA: e_xx, e_yy, e_xy | Blatter-Pattyn: all but dw/dx, dw/dy | "
                               f"du/dz = {sf:g} x (u_s - u_b)/H")])
    if len(ks) > 1:
        layout["sliders"] = [dict(active=len(ks) - 1, currentvalue=dict(prefix=""), pad=dict(t=40),
                                  steps=[dict(label=tl, method="skip", args=[None]) for tl in tlab])]
    fig.update_layout(**layout)
    # the dropdown picks the field, the slider the frame: a small script shows the one map of both (plotly's own
    # buttons can only set a fixed visibility list, which would forget the other control)
    js = """
    var gd = document.getElementById('{plot_id}');
    var nK = %d, nF = %d, labels = %s, tlab = %s, name = %s;
    var cur = {f: 0, k: nK - 1};
    function show() {
      var vis = [];
      for (var k = 0; k < nK; k++) for (var f = 0; f < nF; f++) vis.push(k == cur.k && f == cur.f);
      for (var k = 0; k < nK; k++) vis.push(k == cur.k);
      Plotly.restyle(gd, {visible: vis});
      Plotly.relayout(gd, {'title.text': name + ', ' + tlab[cur.k] + ': ' + labels[cur.f]});
    }
    gd.on('plotly_buttonclicked', function(e) { cur.f = e.active; show(); });
    gd.on('plotly_sliderchange', function(e) { cur.k = e.slider.active; show(); });
    """ % (len(ks), nF, json.dumps(labels), json.dumps(tlab), json.dumps(os.path.basename(path)))
    tag = f"_t{float(times[ks[0]]):.0f}" if len(ks) == 1 else f"_every{every}"
    out = out or os.path.splitext(path)[0] + f"_strain{tag}.html"
    fig.write_html(out, include_plotlyjs="cdn", post_script=js)
    print(f"{len(ks)} frame(s), {nF} maps each -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    try:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out))
    except Exception:
        pass


if __name__ == "__main__":
    main()
