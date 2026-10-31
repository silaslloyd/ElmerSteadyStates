#!/usr/bin/env python3
"""
3D view of states on a footprint several elements wide (e.g. the periodic grounding-line test,
configs/test_periodic_gl.toml): the ice surface Zs (coloured by surface speed), the ice base Zb (coloured
grounded / floating), the bedrock (grey) and the grounding line of every row across y (black), with a
slider through the states.

    python plots/plot_state_3d.py work/test_periodic_gl/fix                 # every chunk of every phase
    python plots/plot_state_3d.py work/test_periodic_gl/fix/retreat_chunks  # one phase
    python plots/plot_state_3d.py STATE.npz [STATE2.npz ...]                # given states
    python plots/plot_state_3d.py work/test_periodic_gl/fix --x 700 820     # x window (km)
    python plots/plot_state_3d.py work/test_periodic_gl/fix --tile 1        # one copy (default 2: side by side, so
                                                                            # anything not periodic shows at the seam)
    python plots/plot_state_3d.py work/test_periodic_gl/fix_loads --no-temp # the velocity panel only
    python plots/plot_state_3d.py work/test_periodic_gl/fix_loads --tmin 255  # temperature colour range from 255 K
    python plots/plot_state_3d.py STATE.npz --fs-ssa 10 [--walls]          # the full-Stokes | SSA boundary (magenta)
    python plots/plot_state_3d.py STATE.npz --full                         # the whole sheet, divide to front
Speed colours max out at the surface speed on the grounding line (--vmax V, or --vmax-all for the largest speed).
Structured footprints and the unstructured triangle ones (configs/*_tri.toml; Delaunay, needs scipy).
With a Temperature field: a second panel next to it, the same geometry coloured by temperature (surface, base and
side walls; the base shows the bed temperature, where a seam stripe from wrong loads would show).
A treatment folder is read phase by phase (spinup, retreat, advance: <phase>_chunks/points/state_*.npz).
Writes <first input>_3d.html and opens it. Needs numpy and plotly.
"""
import glob
import os
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ess.statefile import find_name, load_state  # noqa: E402

# ---- settings
PHASES = ("spinup", "retreat", "advance")
X_WINDOW = None          # (xmin, xmax) km; None: from 60 km behind the most inland grounding line to the front
Z_EXAGGERATION = 1.0     # extra vertical stretch on top of the automatic aspect ratio
ASPECT = dict(x=4.0, y=1.0, z=1.2)
SHOW_BED = False         # the bedrock surface under the ice (--bed to show)
BASE_COLOUR = "speed"    # "speed" (basal speed, the same colour scale as the surface) or "mask" (red grounded, blue floating)
SHOW_MESH = False        # the element edges on the base (--mesh to show)
SIDES = True             # side walls (base to surface) at the window ends and the two outer sides
TILE = 2                 # copies side by side across y (periodicity errors show at the seams); --tile N
VMAX_GL = True           # speed colours up to the surface speed on the grounding line (--vmax-all: the largest)
VMAX = None              # speed colour maximum (m/yr), --vmax V
FS_SSA = 0.0             # km: draw the full-Stokes | SSA boundary this far beyond the grounding line (--fs-ssa 10)
WALLS = False            # walled sides: the distance to the grounding line not wrapped across y (--walls)
TEMP = True              # the temperature panel next to the velocity one (if the states have Temperature); --no-temp
T_RANGE = (None, None)   # temperature colour range (K); None: all temperatures in the window (--tmin / --tmax)
SHOW = True


def collect(args):
    """[(label, path)] in order"""
    out = []
    for a in args:
        if a.endswith(".npz"):
            out.append((os.path.basename(a)[:-4], a))
            continue
        pts = os.path.join(a, "points")
        if os.path.isdir(pts):                   # one phase: <phase>_chunks
            ph = os.path.basename(a.rstrip("/")).replace("_chunks", "")
            out += [(f"{ph} {os.path.basename(f)[6:10]}", f) for f in sorted(glob.glob(os.path.join(pts, "state_*.npz")))]
            continue
        for ph in PHASES:                        # a treatment folder
            out += [(f"{ph} {os.path.basename(f)[6:10]}", f)
                    for f in sorted(glob.glob(os.path.join(a, f"{ph}_chunks", "points", "state_*.npz")))]
    if not out:
        raise SystemExit(f"no states found in {args}")
    return out


def _rec(s, name):
    allf = dict(s.diag)
    allf.update(s.frozen)
    blocks = s.layout.split(s.x)
    for b in s.layout.blocks:
        allf[b.name] = (b.gids, blocks[b.name])
    try:
        return allf[find_name(allf, name)]
    except KeyError:
        return None


def surfaces(path):
    """the footprint grid and the fields on it for one state"""
    s = load_state(path)
    g1, X = _rec(s, "coordinate 1")
    Y = dict(zip(g1.tolist(), _rec(s, "coordinate 2")[1].tolist()))
    Xd = dict(zip(g1.tolist(), X.tolist()))
    u = [_rec(s, f"Velocity {k}") for k in (1, 2, 3)]
    Tr = _rec(s, "Temperature")
    Td = dict(zip(Tr[0].tolist(), Tr[1].tolist())) if Tr is not None else None
    sp = {}
    if all(r is not None for r in u):
        d = [dict(zip(r[0].tolist(), r[1].tolist())) for r in u]
        sp = {g: float(np.sqrt(sum(dk.get(g, 0.0) ** 2 for dk in d))) for g in d[0]}

    def grid(name):
        r = _rec(s, name)
        if r is None:
            return None
        gids, v = r
        x = np.array([Xd[g] for g in gids.tolist()])
        y = np.array([Y[g] for g in gids.tolist()])
        return dict(g=gids, x=x, y=y, v=v)
    top, base = grid("Zs"), grid("Zb")
    gm = _rec(s, "GroundedMask")
    bed = _rec(s, "bedrock")
    if base is not None:
        if gm is not None:
            m = dict(zip(gm[0].tolist(), gm[1].tolist()))
            base["mask"] = np.array([m.get(g, np.nan) for g in base["g"].tolist()])
        if bed is not None:
            b = dict(zip(bed[0].tolist(), bed[1].tolist()))
            base["bed"] = np.array([b.get(g, np.nan) for g in base["g"].tolist()])
        base["speed"] = np.array([sp.get(g, np.nan) for g in base["g"].tolist()])
    if top is not None:
        top["speed"] = np.array([sp.get(g, np.nan) for g in top["g"].tolist()])
    for d in (top, base):
        if d is not None and Td is not None:
            d["T"] = np.array([Td.get(g, np.nan) for g in d["g"].tolist()])
    return top, base, float(s.meta.get("time", np.nan))


def structured(x, y):
    """a structured footprint (rows of equal y, the same x positions in every row), not unstructured triangles"""
    return len(np.unique(np.round(x, 1))) * len(np.unique(np.round(y, 1))) == len(x)


def triangles(x, y):
    """triangles of the footprint: of the rows and columns of a structured one, else Delaunay (the unstructured
    triangle meshes: their footprint is a rectangle, so Delaunay covers it like the mesh)"""
    if not structured(x, y):
        from scipy.spatial import Delaunay
        t = Delaunay(np.c_[x, y] / 1e3).simplices
        return t[:, 0], t[:, 1], t[:, 2]
    ys = np.unique(np.round(y, 1))
    xs = np.unique(np.round(x, 1))
    iy = np.searchsorted(ys, np.round(y, 1))
    ix = np.searchsorted(xs, np.round(x, 1))
    idx = -np.ones((len(xs), len(ys)), int)
    idx[ix, iy] = np.arange(len(x))
    I, J, K = [], [], []
    for a in range(len(xs) - 1):
        for b in range(len(ys) - 1):
            p, q, r, t = idx[a, b], idx[a + 1, b], idx[a + 1, b + 1], idx[a, b + 1]
            if min(p, q, r, t) < 0:
                continue
            I += [p, p]
            J += [q, r]
            K += [r, t]
    return np.array(I), np.array(J), np.array(K)


def contour(x, y, f, level, zs):
    """the level line of the nodal field f on the footprint triangles, as segments (x, y in km, then each of the
    nodal fields zs interpolated onto it), segments separated by None (plotly lines)"""
    i, j, k = triangles(x, y)
    out = [[] for _ in range(2 + len(zs))]
    for t in zip(i, j, k):
        pts = []
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            fa, fb = f[a] - level, f[b] - level
            if (fa < 0) != (fb < 0) and np.isfinite(fa) and np.isfinite(fb):
                w = fa / (fa - fb)
                pts.append([x[a] + w * (x[b] - x[a]), y[a] + w * (y[b] - y[a])]
                           + [z[a] + w * (z[b] - z[a]) for z in zs])
        if len(pts) == 2:
            for c in range(len(out)):
                out[c] += [pts[0][c] / (1e3 if c < 2 else 1), pts[1][c] / (1e3 if c < 2 else 1), None]
    return out


def gl_distance(base, periodic=True):
    """km from the grounding line (the GroundedMask 0 line) for the floating base nodes, 0 for the grounded ones
    (as elmer/GLDistance.F90, the distance wrapped across the periodic sides)"""
    gx, gy = (np.array([v for v in c if v is not None]) for c in contour(base["x"], base["y"], base["mask"], 0.0, [])[:2])
    d = np.zeros(len(base["x"]))
    fl = np.where(base["mask"] < -0.5)[0]
    if not len(gx) or not len(fl):
        return d
    W = (base["y"].max() - base["y"].min()) / 1e3
    pts = np.c_[gx, gy]
    if periodic:
        pts = np.r_[pts, pts + [0, W], pts - [0, W]]
    q = np.c_[base["x"][fl], base["y"][fl]] / 1e3
    try:
        from scipy.spatial import cKDTree
        d[fl] = cKDTree(pts).query(q)[0]
    except ImportError:
        for c in np.array_split(np.arange(len(fl)), max(1, len(fl) // 2000)):
            d[fl[c]] = np.sqrt(((q[c, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(1)
    return d


def gl_x(base):
    """the grounding line's x positions (km)"""
    return np.array([v for v in gl_line(base)[0] if v is not None], float)


def gl_line(base, top=None):
    """per row across y: the last grounded node (mask >= 0): x, y (km), base z and surface z there. On an
    unstructured footprint: the GroundedMask 0 line through the triangles (segments separated by None)."""
    xs, ys, zb, zt = [], [], [], []
    tz = {}
    if top is not None:
        tz = {(round(x, 0), round(y, 0)): v for x, y, v in zip(top["x"], top["y"], top["v"])}
    if not structured(base["x"], base["y"]):
        zt_ = np.array([tz.get((round(x, 0), round(y, 0)), v) for x, y, v in zip(base["x"], base["y"], base["v"])])
        cx, cy, cb, ct = contour(base["x"], base["y"], base["mask"], 0.0, [base["v"], zt_])
        return (np.array(cx, dtype=object), np.array(cy, dtype=object), np.array(cb, dtype=object),
                np.array(ct, dtype=object))
    for yj in np.unique(np.round(base["y"], 1)):
        m = (np.abs(base["y"] - yj) < 1) & (base["mask"] > -0.5)
        if m.any():
            k = np.where(m)[0][np.argmax(base["x"][m])]
            xs.append(base["x"][k] / 1e3)
            ys.append(base["y"][k] / 1e3)
            zb.append(base["v"][k])
            zt.append(tz.get((round(base["x"][k], 0), round(base["y"][k], 0)), base["v"][k]))
    o = np.argsort(ys)
    return np.array(xs)[o], np.array(ys)[o], np.array(zb)[o], np.array(zt)[o]


def main():
    import plotly.graph_objects as go
    global X_WINDOW, TILE, TEMP, T_RANGE, SHOW_MESH, SHOW_BED, FS_SSA, WALLS, VMAX, VMAX_GL
    args = sys.argv[1:]
    if "--vmax-all" in args:
        VMAX_GL = False
        args.remove("--vmax-all")
    if "--vmax" in args:
        k = args.index("--vmax")
        VMAX = float(args[k + 1])
        args = args[:k] + args[k + 2:]
    if "--full" in args:                          # the whole ice sheet, divide to front
        X_WINDOW = (-1.0, 1e9)
        args.remove("--full")
    if "--walls" in args:
        WALLS = True
        args.remove("--walls")
    if "--fs-ssa" in args:                        # --fs-ssa KM: the full-Stokes | SSA boundary
        k = args.index("--fs-ssa")
        FS_SSA = float(args[k + 1])
        args = args[:k] + args[k + 2:]
    if "--bed" in args:
        SHOW_BED = True
        args.remove("--bed")
    if "--mesh" in args:
        SHOW_MESH = True
        args.remove("--mesh")
    if "--no-temp" in args:
        TEMP = False
        args.remove("--no-temp")
    for opt, i in (("--tmin", 0), ("--tmax", 1)):
        if opt in args:
            k = args.index(opt)
            r = list(T_RANGE)
            r[i] = float(args[k + 1])
            T_RANGE = tuple(r)
            args = args[:k] + args[k + 2:]
    if "--x" in args:                             # --x XMIN XMAX (km)
        k = args.index("--x")
        X_WINDOW = (float(args[k + 1]), float(args[k + 2]))
        args = args[:k] + args[k + 3:]
    if "--tile" in args:                          # --tile N copies across y
        k = args.index("--tile")
        TILE = max(1, int(args[k + 1]))
        args = args[:k] + args[k + 2:]
    sys.argv = [sys.argv[0]] + args
    items = collect(args or ["work/test_periodic_gl/fix"])
    print(f"{len(items)} states")
    data = []
    for label, f in items:
        top, base, t = surfaces(f)
        data.append((label, top, base, t))
        if base is None or "mask" not in base:
            print(f"  {label}: no GroundedMask / Zb in {f}")
        else:
            gx = gl_x(base)
            per = (f"per row {np.round(gx, 2)}" if len(gx) <= 6 else
                   f"min {np.min(gx):.2f} mean {np.mean(gx):.2f} max {np.max(gx):.2f} over {len(gx)} rows")
            print(f"  {label}: grounding line {per} km; mask values {np.unique(base['mask'])}")
    # window around the grounding line
    if X_WINDOW:
        x0, x1 = X_WINDOW
    else:
        gls = [gl_x(b) for _, _, b, _ in data if b is not None and "mask" in b]
        gls = [g for g in gls if len(g)]
        xmax = max(float(np.max(top["x"])) for _, top, _, _ in data if top is not None) / 1e3
        x0 = (min(float(np.min(g)) for g in gls if len(g)) - 60.0) if gls else 0.0
        x1 = xmax
    xall = np.concatenate([tp["x"] for _, tp, _, _ in data if tp is not None]) / 1e3
    x0, x1 = max(x0, float(xall.min())), min(x1, float(xall.max()))
    print(f"x window {x0:.0f} .. {x1:.0f} km (domain {xall.min():.0f} .. {xall.max():.0f} km; --x XMIN XMAX to change)")

    def crop(d):
        if d is None:
            return None
        m = (d["x"] / 1e3 >= x0) & (d["x"] / 1e3 <= x1)
        return {k: (v[m] if isinstance(v, np.ndarray) and v.shape == d["x"].shape else v) for k, v in d.items()}
    data = [(l, crop(tp), crop(b), t) for l, tp, b, t in data]
    smax = max(float(np.nanmax(tp["speed"])) for _, tp, _, _ in data if tp is not None)
    if VMAX is not None:
        smax = VMAX
    elif VMAX_GL:                                 # the colour maxes out at the surface speed on the grounding line
        sgl = []
        for _, tp, b, _ in data:
            if tp is None or b is None or "mask" not in b:
                continue
            tzd = {(round(x, 0), round(y, 0)): v for x, y, v in zip(tp["x"], tp["y"], tp["speed"])}
            sp_ = np.array([tzd.get((round(x, 0), round(y, 0)), np.nan) for x, y in zip(b["x"], b["y"])])
            v = [q for q in contour(b["x"], b["y"], b["mask"], 0.0, [sp_])[2] if q is not None and np.isfinite(q)]
            if v:
                sgl.append(max(v))
        if sgl:
            smax = max(sgl)
            print(f"speed colours 0 .. {smax:.0f} m/yr: the largest surface speed on the grounding line (faster ice "
                  f"saturates; --vmax V to set it, --vmax-all for the largest speed)")
    temp = TEMP and all(d is not None and "T" in d for _, tp, b, _ in data for d in (tp, b))
    if TEMP and not temp:
        print("no Temperature in the states (slimmed before it was kept? re-run plots/slim_states.py): velocity only")
    if temp:
        Tv = np.concatenate([d["T"] for _, tp, b, _ in data for d in (tp, b)])   # the full range, surface and base
        tmin = T_RANGE[0] if T_RANGE[0] is not None else float(np.nanmin(Tv))
        tmax = T_RANGE[1] if T_RANGE[1] is not None else float(np.nanmax(Tv))
        print(f"temperature colours {tmin:.2f} .. {tmax:.2f} K: all temperatures (--tmin / --tmax to narrow)")
    # the panels: (field, colour range, colour scale, colour bar title, colour bar x, scene)
    panels = [("speed", 0.0, smax, "Viridis", "speed (m/yr)", 0.47 if temp else 1.0, "scene")]
    if temp:
        panels.append(("T", tmin, tmax, "RdBu_r", "temperature (K)", 1.0, "scene2"))

    W = max(float(np.ptp(tp["y"])) for _, tp, _, _ in data if tp is not None)      # strip width (m)

    def mesh_lines(base, P):
        """the element edges of the base (rows along x, columns along y), just below it, for every copy"""
        xr, yr = np.round(base["x"], 0), np.round(base["y"], 0)
        lx, ly, lz = [], [], []
        for c in range(TILE):
            dy = c * W / 1e3
            for key, other in ((yr, xr), (xr, yr)):        # lines of constant y (along x), then of constant x
                for v in np.unique(key):
                    m = np.where(key == v)[0]
                    o = m[np.argsort(other[m])]
                    lx += list(base["x"][o] / 1e3) + [None]
                    ly += list(base["y"][o] / 1e3 + dy) + [None]
                    lz += list(base["v"][o] - 3.0) + [None]
        return go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="#222222", width=1),
                            name="base mesh", hoverinfo="skip", scene=P[6], showlegend=P[6] == "scene")

    def walls(top, base, P):
        """side walls between the base and the surface: the x-window ends of every copy and the two outer y sides,
        coloured by speed (surface speed at the top, basal speed at the bottom); not at the seams"""
        f, lo, hi, cs = P[0], P[1], P[2], P[3]
        b = {(round(x, 0), round(y, 0)): (z, sp) for x, y, z, sp in zip(base["x"], base["y"], base["v"], base[f])}
        out = []

        def wall(idx, dy, name):
            idx = [k for k in idx if (round(top["x"][k], 0), round(top["y"][k], 0)) in b]
            if len(idx) < 2:
                return
            n = len(idx)
            X = np.r_[top["x"][idx], top["x"][idx]] / 1e3
            Y = np.r_[top["y"][idx], top["y"][idx]] / 1e3 + dy
            zb_, sb_ = zip(*[b[(round(top["x"][k], 0), round(top["y"][k], 0))] for k in idx])
            Z = np.r_[top["v"][idx], np.array(zb_)]
            S = np.r_[top[f][idx], np.array(sb_)]
            # segment a: top a, a + 1 and bottom n + a, n + a + 1 -> triangles (a, a+1, n+a), (a+1, n+a+1, n+a)
            I = [v for a in range(n - 1) for v in (a, a + 1)]
            J = [v for a in range(n - 1) for v in (a + 1, n + a + 1)]
            K = [v for a in range(n - 1) for v in (n + a, n + a)]
            out.append(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, intensity=S, colorscale=cs, cmin=lo, cmax=hi,
                                 showscale=False, opacity=1.0, name=name, showlegend=False, hoverinfo="skip",
                                 scene=P[6]))
        xs, ys = top["x"], top["y"]
        for c in range(TILE):
            dy = c * W / 1e3
            for xe in (xs.min(), xs.max()):                       # the x ends of the window
                m = np.where(np.abs(xs - xe) < 1)[0]
                wall(m[np.argsort(ys[m])], dy, "end wall")
        ylo, yhi = ys.min(), ys.max()
        for ye, c in ((ylo, 0), (yhi, TILE - 1)):                 # the outer sides
            m = np.where(np.abs(ys - ye) < 1)[0]
            wall(m[np.argsort(xs[m])], c * W / 1e3, "side wall")
        return out

    def traces(top, base):
        return [t for P in panels for t in panel_traces(top, base, P)]

    def tiled(d):
        """TILE copies of a surface, copy c shifted by c W in y, as one node set: each seam row (y = W of one copy,
        y = 0 of the next: the same values on a periodic solution) once, so the copies share it"""
        if d is None:
            return None
        keys = [k for k, v in d.items() if isinstance(v, np.ndarray) and v.shape == d["x"].shape]
        out = {k: np.concatenate([d[k] + (c * W if k == "y" else 0) for c in range(TILE)]) for k in keys}
        _, first = np.unique(np.c_[np.round(out["x"], 0), np.round(out["y"], 0)], axis=0, return_index=True)
        first = np.sort(first)
        return {k: v[first] for k, v in out.items()}

    def panel_traces(top, base, P):
        """TILE copies side by side, copy c shifted by c W in y: at each seam y = c W the image side (y = W) of
        one copy meets the master side (y = 0) of the next, so anything not periodic shows as a step there"""
        f, lo, hi, cs, ctitle, cx, sc = P
        unit = "m/yr" if f == "speed" else "K"
        first_panel = sc == "scene"
        tr = []
        # all copies as ONE surface each (the seam rows shared): separate surfaces are lit with one-sided normals at
        # their edges, which draws a faint line along a seam even where the values match exactly
        T_, B_ = tiled(top), tiled(base)
        if T_ is not None:
            i, j, k = triangles(T_["x"], T_["y"])
            tr.append(go.Mesh3d(x=T_["x"] / 1e3, y=T_["y"] / 1e3, z=T_["v"], i=i, j=j, k=k,
                                intensity=T_[f], colorscale=cs, cmin=lo, cmax=hi, showscale=True,
                                colorbar=dict(title=ctitle, x=cx, len=0.8), name="surface Zs",
                                showlegend=False, customdata=T_[f], scene=sc,
                                hovertemplate="x %{x:.2f} km, y %{y:.2f} km<br>Zs %{z:.1f} m<br>"
                                              f"{f} %{{customdata:.2f}} {unit}<extra></extra>"))
        if B_ is not None:
            i, j, k = triangles(B_["x"], B_["y"])
            mask = B_.get("mask", np.zeros_like(B_["v"]))
            if f == "T" or BASE_COLOUR == "speed":
                col = dict(intensity=B_[f], colorscale=cs, cmin=lo, cmax=hi, showscale=False)
            else:
                col = dict(intensity=np.where(mask > -0.5, 1.0, 0.0), cmin=0, cmax=1,
                           colorscale=[[0, "#3a7bd5"], [1, "#c0392b"]], showscale=False)
            tr.append(go.Mesh3d(x=B_["x"] / 1e3, y=B_["y"] / 1e3, z=B_["v"], i=i, j=j, k=k, **col,
                                customdata=np.stack([B_[f], mask], 1), scene=sc,
                                name="base Zb", opacity=1.0, showlegend=False,
                                hovertemplate="x %{x:.2f} km, y %{y:.2f} km<br>Zb %{z:.1f} m<br>"
                                              f"{'speed' if f == 'speed' else 'bed T'} %{{customdata[0]:.2f}} {unit}"
                                              "<br>mask %{customdata[1]}<extra></extra>"))
            if SHOW_BED and "bed" in B_:
                tr.append(go.Mesh3d(x=B_["x"] / 1e3, y=B_["y"] / 1e3, z=B_["bed"] - 1.0, i=i, j=j,
                                    k=k, color="#999999", opacity=0.35, name="bedrock", hoverinfo="skip",
                                    showlegend=False, scene=sc))
        if SHOW_MESH and base is not None:
            tr.append(mesh_lines(base, P))
        if SIDES and top is not None and base is not None:
            tr += walls(top, base, P)
        if base is not None and "mask" in base:
            # the grounding line through all copies (a jump at a seam: not periodic), in red on the surface and on
            # the base (just outside each, so neither surface hides it)
            gx, gy, gzb, gzt = gl_line(base, top)
            ax, ay, at, ab = [], [], [], []
            add = lambda v, a: None if v is None else v + a
            for c in range(TILE):
                ax += list(gx) + [None]
                ay += [add(v, c * W / 1e3) for v in gy] + [None]
                at += [add(v, 5.0) for v in gzt] + [None]
                ab += [add(v, -5.0) for v in gzb] + [None]
            unstr = not structured(base["x"], base["y"])
            for z_, nm in ((at, "grounding line (surface)"), (ab, "grounding line (base)")):
                tr.append(go.Scatter3d(x=ax, y=ay, z=z_, mode="lines" if unstr else "lines+markers",
                                       line=dict(color="red", width=8), marker=dict(size=3, color="red"), name=nm,
                                       scene=sc, showlegend=first_panel, connectgaps=False,
                                       hovertemplate="GL x %{x:.2f} km at y %{y:.2f} km<extra></extra>"))
            if FS_SSA:
                # the boundary between the full-Stokes region and the SSA shelf: floating ice more than FS_SSA km from
                # the grounding line is the passive SSA shelf (elmer/GLDistance.F90, FSSSAShelf); magenta on the
                # surface and the base, and a curtain through the ice
                tzd = {(round(x, 0), round(y, 0)): v for x, y, v in zip(top["x"], top["y"], top["v"])} if top else {}
                zt_ = np.array([tzd.get((round(x, 0), round(y, 0)), v) for x, y, v in
                                zip(base["x"], base["y"], base["v"])])
                dist = gl_distance(base, periodic=not WALLS)
                cx, cy, cb, ct = contour(base["x"], base["y"], np.where(base["mask"] < -0.5, dist, 0.0), FS_SSA,
                                         [base["v"], zt_])
                bx, by, btop, bbot = [], [], [], []
                for c in range(TILE):
                    bx += cx + [None]
                    by += [add(v, c * W / 1e3) for v in cy] + [None]
                    btop += [add(v, 5.0) for v in ct] + [None]
                    bbot += [add(v, -5.0) for v in cb] + [None]
                nm = f"full-Stokes | SSA boundary ({FS_SSA:g} km beyond the GL)"
                for z_, show in ((btop, first_panel), (bbot, False)):
                    tr.append(go.Scatter3d(x=bx, y=by, z=z_, mode="lines", line=dict(color="magenta", width=9),
                                           name=nm, legendgroup="fsssa", scene=sc, showlegend=show, connectgaps=False,
                                           hovertemplate="FS | SSA x %{x:.2f} km at y %{y:.2f} km<extra></extra>"))
                # the curtain: each segment a quad from the base to the surface
                X, Y, Z, I, J, K = [], [], [], [], [], []
                for q in range(0, len(bx) - 1):
                    if bx[q] is None or bx[q + 1] is None:
                        continue
                    n0 = len(X)
                    X += [bx[q], bx[q], bx[q + 1], bx[q + 1]]
                    Y += [by[q], by[q], by[q + 1], by[q + 1]]
                    Z += [bbot[q], btop[q], bbot[q + 1], btop[q + 1]]
                    I += [n0, n0 + 1]; J += [n0 + 1, n0 + 3]; K += [n0 + 2, n0 + 2]
                if X:
                    tr.append(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color="magenta", opacity=0.35, name=nm,
                                        legendgroup="fsssa", showlegend=False, hoverinfo="skip", scene=sc))
                if first_panel:
                    px = [v for v in cx if v is not None]
                    if px:
                        print(f"  full-Stokes | SSA boundary: x {min(px):.2f} .. {max(px):.2f} km")
        if top is not None and TILE > 1:
            m = np.abs(top["y"] - W) < 1
            o = np.argsort(top["x"][m])
            sx, sz = top["x"][m][o] / 1e3, top["v"][m][o] + 15.0
            qx, qy, qz = [], [], []
            for c in range(1, TILE):
                qx += list(sx) + [None]
                qy += [c * W / 1e3] * len(sx) + [None]
                qz += list(sz) + [None]
            tr.append(go.Scatter3d(x=qx, y=qy, z=qz, mode="lines",
                                   line=dict(color="white" if f == "speed" else "black", width=3, dash="dash"),
                                   name="periodic seam (y = W of one copy = y = 0 of the next)", hoverinfo="skip",
                                   scene=sc, showlegend=first_panel))
        return tr
    first = traces(data[0][1], data[0][2])
    frames = [go.Frame(data=traces(tp, b), name=str(n)) for n, (_, tp, b, _) in enumerate(data)]
    steps = [dict(method="animate", label=l, args=[[str(n)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                                  transition=dict(duration=0))])
             for n, (l, _, _, _) in enumerate(data)]
    zs = [v for _, tp, b, _ in data for d in (tp, b) if d is not None for v in (np.nanmin(d["v"]), np.nanmax(d["v"]))]
    zs += [np.nanmin(b["bed"]) for _, _, b, _ in data if b is not None and "bed" in b]
    asp = dict(ASPECT)
    asp["z"] = ASPECT["z"] * Z_EXAGGERATION
    asp["y"] = ASPECT["y"] * TILE
    fig = go.Figure(data=first, frames=frames)
    scene = dict(xaxis=dict(title="x (km)", range=[x0, x1]), yaxis=dict(title="y (km)"),
                 zaxis=dict(title="z (m)", range=[min(zs) - 20, max(zs) + 20]),
                 aspectmode="manual", aspectratio=asp)
    scenes = {"scene": dict(scene, domain=dict(x=[0.0, 0.46] if temp else [0.0, 1.0], y=[0.0, 1.0]))}
    if temp:
        scenes["scene2"] = dict(scene, domain=dict(x=[0.52, 0.98], y=[0.0, 1.0]))
    ann = ([dict(text="velocity", x=0.23, y=1.0, xref="paper", yref="paper", showarrow=False, font=dict(size=16)),
            dict(text="temperature", x=0.75, y=1.0, xref="paper", yref="paper", showarrow=False, font=dict(size=16))]
           if temp else [])
    fig.update_layout(
        title=f"{sys.argv[1] if len(sys.argv) > 1 else ''}: {data[0][0]}", **scenes, annotations=ann,
        sliders=[dict(steps=steps, currentvalue=dict(prefix="state: "), pad=dict(t=30))],
        autosize=True, template="plotly_white", legend=dict(x=0, y=1),
        margin=dict(l=10, r=10, t=60, b=10))
    out = os.path.splitext(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "plot_state_3d")[0] + "_3d.html"
    sync = ("""
var gd = document.getElementsByClassName('plotly-graph-div')[0], busy = false;
gd.on('plotly_relayout', function (e) {
  if (busy) return;
  var u = {};
  if (e['scene.camera']) u['scene2.camera'] = e['scene.camera'];
  if (e['scene2.camera']) u['scene.camera'] = e['scene2.camera'];
  if (Object.keys(u).length) { busy = true; Plotly.relayout(gd, u).then(function () { busy = false; }); }
});""" if temp else None)
    # the page width and height (no fixed size: no sideways scrolling; resizes with the window)
    fig.write_html(out, include_plotlyjs="cdn", post_script=sync, config=dict(responsive=True),
                   default_width="100%", default_height="95vh")
    print(f"wrote {out}")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
