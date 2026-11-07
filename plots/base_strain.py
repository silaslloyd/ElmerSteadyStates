#!/usr/bin/env python3
"""
Strain rates at the base, map view (plotly), from a frames file (plots/vtu_frames.py): which terms dominate where,
and what each approximation (SIA, SSA, Blatter-Pattyn, full Stokes) keeps and drops.

    python3 plots/base_strain.py FRAMES.npz [FRAMES2.npz ...] [--every N | --frame K | --t YEARS] [--res KM] [--x XMIN XMAX]
        [--yscale S] [--shear-factor F] [--out FILE.html]

Default: every 5th frame (--every N; the last always; --first N: only the first N frames), a slider through them and a dropdown for the map; --frame /
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
and the dominant term at each point. The grounding line in black. "base" / "surface" switches between the horizontal
derivatives of the base velocity and of the surface velocity (the vertical shear estimate is the same column mean on
both; at the free surface the true shear is near zero).

Blatter-Pattyn's second approximation, the hydrostatic vertical stress, drops the "bridging" term: sigma_zz(base) +
rho g H = d/dx int tau_xz dz + d/dy int tau_yz dz. With the shallow-ice shear stress the integrals are
-rho g H^2/2 grad s, so the map "non-hydrostatic estimate" = |div(H^2 grad s)| / (2H), as a fraction of the
overburden rho g H: well below 1% (blue), the hydrostatic assumption holds; red, it does not. An estimate from the
geometry (the frames hold no stress): a check of the model's own stress needs the basal normal stress in the output.

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


def frame_fields(d, k, x, y, tri, sf, interp_top, grounded_only=True, side="base"):
    """the maps of frame k on the base or the surface (at the base nodes' x, y): [(label, values, kind)], kind: signed |
    log | share | ratio | nh | cat"""
    ub, vb, wb = (d[f"base_v{c}"][k].astype(float) for c in "xyz")
    us, vs, ws, H, s = interp_top(k)
    u, v, w = (ub, vb, wb) if side == "base" else (us, vs, ws)
    # grounded only: the grounding line (GroundedMask 0) and the shelf (-1) masked out, and the gradients from the
    # grounded triangles only (not across the jump at the grounding line)
    gm = d["base_mask"][k].astype(float)
    use = gm > 0.5 if grounded_only else None
    g = node_gradients(tri, x, y, {"u": u, "v": v, "w": w}, use)
    dudx, dudy = g["u"]; dvdx, dvdy = g["v"]; dwdx, dwdy = g["w"]
    with np.errstate(invalid="ignore", divide="ignore"):
        Hs = np.where(H > 1.0, H, np.nan)
        dudz, dvdz = sf * (us - ub) / Hs, sf * (vs - vb) / Hs
    exx, eyy = dudx, dvdy
    exy = 0.5 * (dudy + dvdx)
    exz, eyz = 0.5 * (dudz + dwdx), 0.5 * (dvdz + dwdy)
    memb = np.sqrt(np.maximum(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2, 0))
    shear = np.sqrt(exz ** 2 + eyz ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mshare = memb / (memb + shear)
        sh = np.hypot(dudz, dvdz)
        wratio = np.where(sh > 0, np.hypot(dwdx, dwdy) / sh, np.nan)     # no vertical shear (plug flow): undefined
    # Blatter-Pattyn's second approximation, the hydrostatic vertical stress: what it drops is the "bridging" term,
    # sigma_zz(base) + rho g H = d/dx int tau_xz dz + d/dy int tau_yz dz. With the shallow-ice shear stress (tau_xz =
    # -rho g (s - z) ds/dx) the integrals are -rho g H^2/2 grad s, so relative to the overburden rho g H:
    #     delta = |div(H^2 grad s)| / (2 H)     (an estimate: << 1, hydrostatic holds; ~0.1 or more, it does not)
    gs = node_gradients(tri, x, y, {"s": s}, use)
    with np.errstate(invalid="ignore"):
        Fx, Fy = H ** 2 * gs["s"][0], H ** 2 * gs["s"][1]
    gf = node_gradients(tri, x, y, {"Fx": Fx, "Fy": Fy}, use)
    with np.errstate(invalid="ignore", divide="ignore"):
        nh = np.abs(gf["Fx"][0] + gf["Fy"][1]) / (2 * Hs)
    A = np.abs(np.c_[tuple(a for _, a in dominant_terms(exx, eyy, exy, dudz, dvdz, dwdx, dwdy))])
    dom = np.where(np.isfinite(A).all(axis=1), np.argmax(np.nan_to_num(A, nan=-1), axis=1), -1).astype(float)
    dom[dom < 0] = np.nan
    if grounded_only:
        off = ~use
        for a in (exx, eyy, exy, exz, eyz, dudz, dvdz, dwdx, dwdy, dudy, dvdx, memb, shear, mshare, wratio, dom, nh):
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
            ("log10 non-hydrostatic estimate |div(H^2 grad s)| / 2H (Blatter-Pattyn's hydrostatic assumption)",
             np.log10(np.maximum(nh, 1e-8)), "nh"),
            ("dominant term", dom, "cat")]


def dominant_terms(exx, eyy, exy, dudz, dvdz, dwdx, dwdy):
    return [("e_xx", exx), ("e_yy", eyy), ("e_xy", exy), ("du/dz, dv/dz / 2", 0.5 * np.hypot(dudz, dvdz)),
            ("dw/dx, dw/dy / 2", 0.5 * np.hypot(dwdx, dwdy))]


class Joined:
    """several frames files as one: the frames in the order of the files (a later file's frames at or before the
    previous one's last time are dropped), on the first file's nodes (a file on other nodes, e.g. another
    partitioning of the same mesh, is mapped node by node by position)"""
    STATIC = ("base_x", "base_y", "top_x", "top_y")

    def __init__(self, paths):
        from scipy.spatial import cKDTree
        self.files = [np.load(p, allow_pickle=True) for p in paths]
        ref = self.files[0]
        self.idx, self.ref = [], ref
        frames, tmax = [], -np.inf
        for i, f in enumerate(self.files):
            maps = {}
            for side in ("base", "top"):
                if i == 0 or (len(f[f"{side}_x"]) == len(ref[f"{side}_x"]) and np.allclose(f[f"{side}_x"], ref[f"{side}_x"])
                              and np.allclose(f[f"{side}_y"], ref[f"{side}_y"])):
                    maps[side] = None
                else:
                    dist, j = cKDTree(np.c_[f[f"{side}_x"], f[f"{side}_y"]]).query(np.c_[ref[f"{side}_x"], ref[f"{side}_y"]])
                    if dist.max() > 1.0:
                        raise SystemExit(f"{paths[i]}: not the same mesh as {paths[0]} (nodes {dist.max():.0f} m apart)")
                    maps[side] = j
            self.idx.append(maps)
            for j, t in enumerate(f["frame_times"]):
                if t > tmax + 1e-6:
                    frames.append((i, j)); tmax = float(t)
        self.frames = frames
        self.times = np.array([float(self.files[i]["frame_times"][j]) for i, j in frames])

    def __getitem__(self, key):
        if key in self.STATIC:
            return self.ref[key]
        if key == "frame_times":
            return self.times
        side = "base" if key.startswith("base_") else "top"
        outer = self

        class Per:
            def __getitem__(self, k):
                i, j = outer.frames[k]
                a = outer.files[i][key]
                a = a[j] if a.ndim == 2 else a
                m = outer.idx[i][side]
                return a if m is None else a[m]
        return Per()


def main():
    args = sys.argv[1:]
    args, frame = opt(args, "--frame", conv=int)
    args, tsel = opt(args, "--t")
    args, every = opt(args, "--every", conv=int, default=5)
    args, first = opt(args, "--first", conv=int)           # --first N: only the first N frames
    args, res = opt(args, "--res")
    args, xwin = opt(args, "--x", n=2)
    args, yscale = opt(args, "--yscale", default=3.0)
    args, sf = opt(args, "--shear-factor", default=1.0)
    args, out = opt(args, "--out", conv=str)
    grounded_only = "--keep-shelf" not in args            # --keep-shelf: the grounding line and the shelf too
    args = [a for a in args if a != "--keep-shelf"]
    if not args:
        raise SystemExit(__doc__)
    paths = args
    path = paths[-1] if len(paths) == 1 else paths[0]
    from scipy.spatial import Delaunay, cKDTree
    from scipy.interpolate import LinearNDInterpolator
    import plotly.graph_objects as go

    d = Joined(paths)
    times = d["frame_times"]
    name = " + ".join(os.path.basename(p) for p in paths)
    if frame is not None or tsel is not None:      # one frame
        ks = [int(np.argmin(np.abs(times - tsel)))] if tsel is not None else [frame % len(times)]
    else:                                          # every N-th frame, and the last
        nt = min(first, len(times)) if first else len(times)
        ks = sorted(set(list(range(0, nt, every)) + [nt - 1]))
    res = res or (1.0 if len(ks) == 1 else 2.0)    # km; coarser with a slider (the page holds every map of every frame)
    x, y = d["base_x"].astype(float), d["base_y"].astype(float)
    tx, ty = d["top_x"].astype(float), d["top_y"].astype(float)
    same = len(tx) == len(x) and np.allclose(tx, x) and np.allclose(ty, y)
    tt = None if same else Delaunay(np.c_[tx, ty])

    def interp_top(k):                             # the surface velocity, thickness and elevation at the base nodes
        if same:
            f = lambda a: a.astype(float)
        else:
            f = lambda a: LinearNDInterpolator(tt, a.astype(float))(x, y)
        s = f(d["top_z"][k])
        return f(d["top_vx"][k]), f(d["top_vy"][k]), f(d["top_vz"][k]), s - d["base_z"][k].astype(float), s

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

    sides = ["base", "surface"]
    maps, glz = [], []                             # maps[side * nK + frame][field] = (label, grid, kind)
    for side in sides:
        for n, k in enumerate(ks):
            maps.append([(lab, grid(f, kind == "cat"), kind)
                         for lab, f, kind in frame_fields(d, k, x, y, tri, sf, interp_top, grounded_only, side)])
            print(f"  {side} frame {k} (t {float(times[k]):g} yr) {n + 1}/{len(ks)}", end="\r", flush=True)
    for k in ks:
        glz.append(grid(d["base_mask"][k].astype(float)))
    print()
    nF = len(maps[0])
    labels = [m[0] for m in maps[0]]
    # one colour range per kind of map, over all the frames shown and all the maps of that kind: every strain-rate
    # component and derivative on the same scale (so their sizes compare directly), and the two log maps on one
    def pooled(kind):
        z = np.concatenate([m[j][1].ravel() for m in maps for j in range(nF) if maps[0][j][2] == kind])
        return z[np.isfinite(z)]
    zs = pooled("signed")
    a_signed = float(np.percentile(np.abs(zs), 99)) if zs.size else 1.0
    zl = pooled("log")
    lo_log, hi_log = (float(v) for v in np.percentile(zl, [1, 99.5])) if zl.size else (-6.0, 0.0)
    rng = []
    for j in range(nF):
        kind = maps[0][j][2]
        if kind == "signed":
            rng.append(dict(colorscale="RdBu_r", zmid=0, zmin=-a_signed, zmax=a_signed, colorbar=dict(title="1/yr")))
        elif kind == "log":
            rng.append(dict(colorscale="Viridis", zmin=lo_log, zmax=hi_log, colorbar=dict(title="log10 1/yr")))
        elif kind == "share":
            rng.append(dict(colorscale="Cividis", zmin=0, zmax=1, colorbar=dict(title="0 SIA .. 1 SSA")))
        elif kind == "ratio":
            rng.append(dict(colorscale="RdBu_r", zmid=0, zmin=-3, zmax=1, colorbar=dict(title="log10 ratio")))
        elif kind == "nh":                         # 1e-4 .. 1 of the overburden; red above 1e-2
            rng.append(dict(colorscale="RdBu_r", zmid=-2, zmin=-4, zmax=0,
                            colorbar=dict(title="log10 of rho g H", tickvals=[-4, -3, -2, -1, 0],
                                          ticktext=["1e-4", "1e-3", "1%", "10%", "100%"])))
        else:
            names = [n_ for n_, _ in dominant_terms(*([np.zeros(1)] * 7))]
            cols = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e"]
            cs = []
            for i, c in enumerate(cols[:len(names)]):
                cs += [[i / len(names), c], [(i + 1) / len(names), c]]
            rng.append(dict(colorscale=cs, zmin=-0.5, zmax=len(names) - 0.5,
                            colorbar=dict(title="dominant", tickvals=list(range(len(names))), ticktext=names)))

    nK = len(ks)
    traces = []
    for n, m in enumerate(maps):                   # side-major, then frame: trace (side * nK + frame) * nF + field
        for j, (lab, Z, kind) in enumerate(m):
            traces.append(go.Heatmap(x=gx, y=gy, z=Z, visible=(n == nK - 1 and j == 0), **rng[j],
                                     hovertemplate=f"{lab}: %{{z:.3g}}<br>x %{{x:.1f}} y %{{y:.1f}} km<extra></extra>"))
    for n, Z in enumerate(glz):                    # then the grounding line of each frame
        traces.append(go.Contour(x=gx, y=gy, z=Z, contours=dict(start=-0.5, end=-0.5, size=1, coloring="none"),
                                 line=dict(color="black", width=2), showscale=False, hoverinfo="skip",
                                 visible=(n == nK - 1), name="grounding line"))
    tlab = [f"t {float(times[k]):,.0f} yr" for k in ks]
    fig = go.Figure(traces)
    layout = dict(
        title=f"{name}, base, {tlab[-1]}: {labels[0]}",
        updatemenus=[dict(buttons=[dict(label=lab, method="skip", args=[None]) for lab in labels], direction="down",
                          x=0.09, y=1.12, xanchor="left", yanchor="bottom", active=0),
                     dict(buttons=[dict(label=sd, method="skip", args=[None]) for sd in sides], type="buttons",
                          direction="right", x=0, y=1.12, xanchor="left", yanchor="bottom", active=0, showactive=True)],
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
    var nK = %d, nF = %d, labels = %s, tlab = %s, name = %s, sides = %s;
    var cur = {s: 0, f: 0, k: nK - 1};
    function show() {
      var vis = [];
      for (var s = 0; s < sides.length; s++)
        for (var k = 0; k < nK; k++) for (var f = 0; f < nF; f++) vis.push(s == cur.s && k == cur.k && f == cur.f);
      for (var k = 0; k < nK; k++) vis.push(k == cur.k);
      Plotly.restyle(gd, {visible: vis});
      Plotly.relayout(gd, {'title.text': name + ', ' + sides[cur.s] + ', ' + tlab[cur.k] + ': ' + labels[cur.f]});
    }
    gd.on('plotly_buttonclicked', function(e) {
      var i = sides.indexOf(e.button.label);
      if (i >= 0) cur.s = i; else cur.f = labels.indexOf(e.button.label);
      show();
    });
    gd.on('plotly_sliderchange', function(e) { cur.k = e.slider.active; show(); });
    """ % (len(ks), nF, json.dumps(labels), json.dumps(tlab), json.dumps(name), json.dumps(sides))
    tag = f"_t{float(times[ks[0]]):.0f}" if len(ks) == 1 else f"_every{every}" + (f"_first{first}" if first else "")
    out = out or os.path.splitext(paths[-1])[0] + ("_joined" if len(paths) > 1 else "") + f"_strain{tag}.html"
    fig.write_html(out, include_plotlyjs="cdn", post_script=js)
    print(f"{len(ks)} frame(s), {nF} maps each -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    try:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out))
    except Exception:
        pass


if __name__ == "__main__":
    main()
