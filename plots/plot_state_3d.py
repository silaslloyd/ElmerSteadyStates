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
SHOW_BED = True
BASE_COLOUR = "speed"    # "speed" (basal speed, the same colour scale as the surface) or "mask" (red grounded, blue floating)
SHOW_MESH = True         # the element edges on the base
SIDES = True             # side walls (base to surface) at the window ends and the two outer sides
TILE = 2                 # copies side by side across y (periodicity errors show at the seams); --tile N
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
    return top, base, float(s.meta.get("time", np.nan))


def triangles(x, y):
    """triangles of the structured footprint: rows of equal y, the same x positions in every row"""
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


def gl_line(base, top=None):
    """per row across y: the last grounded node (mask >= 0): x, y (km), base z and surface z there"""
    xs, ys, zb, zt = [], [], [], []
    tz = {}
    if top is not None:
        tz = {(round(x, 0), round(y, 0)): v for x, y, v in zip(top["x"], top["y"], top["v"])}
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
    global X_WINDOW, TILE
    args = sys.argv[1:]
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
            gx = gl_line(base)[0]
            per = (f"per row {np.round(gx, 2)}" if len(gx) <= 6 else
                   f"min {np.min(gx):.2f} mean {np.mean(gx):.2f} max {np.max(gx):.2f} over {len(gx)} rows")
            print(f"  {label}: grounding line {per} km; mask values {np.unique(base['mask'])}")
    # window around the grounding line
    if X_WINDOW:
        x0, x1 = X_WINDOW
    else:
        gls = [gl_line(b)[0] for _, _, b, _ in data if b is not None and "mask" in b]
        gls = [g for g in gls if len(g)]
        xmax = max(float(np.max(top["x"])) for _, top, _, _ in data if top is not None) / 1e3
        x0 = (min(float(np.min(g)) for g in gls if len(g)) - 60.0) if gls else 0.0
        x1 = xmax
    xall = np.concatenate([tp["x"] for _, tp, _, _ in data if tp is not None]) / 1e3
    print(f"x window {x0:.0f} .. {x1:.0f} km (domain {xall.min():.0f} .. {xall.max():.0f} km; --x XMIN XMAX to change)")

    def crop(d):
        if d is None:
            return None
        m = (d["x"] / 1e3 >= x0) & (d["x"] / 1e3 <= x1)
        return {k: (v[m] if isinstance(v, np.ndarray) and v.shape == d["x"].shape else v) for k, v in d.items()}
    data = [(l, crop(tp), crop(b), t) for l, tp, b, t in data]
    smax = max(float(np.nanmax(tp["speed"])) for _, tp, _, _ in data if tp is not None)

    W = max(float(np.ptp(tp["y"])) for _, tp, _, _ in data if tp is not None)      # strip width (m)

    def mesh_lines(base):
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
                            name="base mesh", hoverinfo="skip")

    def walls(top, base):
        """side walls between the base and the surface: the x-window ends of every copy and the two outer y sides,
        coloured by speed (surface speed at the top, basal speed at the bottom); not at the seams"""
        b = {(round(x, 0), round(y, 0)): (z, sp) for x, y, z, sp in zip(base["x"], base["y"], base["v"], base["speed"])}
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
            S = np.r_[top["speed"][idx], np.array(sb_)]
            # segment a: top a, a + 1 and bottom n + a, n + a + 1 -> triangles (a, a+1, n+a), (a+1, n+a+1, n+a)
            I = [v for a in range(n - 1) for v in (a, a + 1)]
            J = [v for a in range(n - 1) for v in (a + 1, n + a + 1)]
            K = [v for a in range(n - 1) for v in (n + a, n + a)]
            out.append(go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, intensity=S, colorscale="Viridis", cmin=0, cmax=smax,
                                 showscale=False, opacity=1.0, name=name, showlegend=False, hoverinfo="skip"))
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
        """TILE copies side by side, copy c shifted by c W in y: at each seam y = c W the image side (y = W) of
        one copy meets the master side (y = 0) of the next, so anything not periodic shows as a step there"""
        tr = []
        for c in range(TILE):
            dy = c * W / 1e3
            leg = c == 0
            if top is not None:
                i, j, k = triangles(top["x"], top["y"])
                tr.append(go.Mesh3d(x=top["x"] / 1e3, y=top["y"] / 1e3 + dy, z=top["v"], i=i, j=j, k=k,
                                    intensity=top["speed"], colorscale="Viridis", cmin=0, cmax=smax, showscale=leg,
                                    colorbar=dict(title="surface speed (m/yr)", x=1.0), name="surface Zs",
                                    showlegend=False,
                                    hovertemplate="x %{x:.2f} km, y %{y:.2f} km<br>Zs %{z:.1f} m<extra></extra>"))
            if base is not None:
                i, j, k = triangles(base["x"], base["y"])
                mask = base.get("mask", np.zeros_like(base["v"]))
                if BASE_COLOUR == "speed":
                    col = dict(intensity=base["speed"], colorscale="Viridis", cmin=0, cmax=smax, showscale=False)
                else:
                    col = dict(intensity=np.where(mask > -0.5, 1.0, 0.0), cmin=0, cmax=1,
                               colorscale=[[0, "#3a7bd5"], [1, "#c0392b"]], showscale=False)
                tr.append(go.Mesh3d(x=base["x"] / 1e3, y=base["y"] / 1e3 + dy, z=base["v"], i=i, j=j, k=k, **col,
                                    customdata=np.stack([base["speed"], mask], 1),
                                    name="base Zb", opacity=1.0, showlegend=False,
                                    hovertemplate="x %{x:.2f} km, y %{y:.2f} km<br>Zb %{z:.1f} m<br>speed "
                                                  "%{customdata[0]:.1f} m/yr<br>mask %{customdata[1]}<extra></extra>"))
                if SHOW_BED and "bed" in base:
                    tr.append(go.Mesh3d(x=base["x"] / 1e3, y=base["y"] / 1e3 + dy, z=base["bed"] - 1.0, i=i, j=j,
                                        k=k, color="#999999", opacity=0.35, name="bedrock", hoverinfo="skip",
                                        showlegend=False))
        if SHOW_MESH and base is not None:
            tr.append(mesh_lines(base))
        if SIDES and top is not None and base is not None:
            tr += walls(top, base)
        if base is not None and "mask" in base:
            # the grounding line through all copies (a jump at a seam: not periodic), in red on the surface and on
            # the base (just outside each, so neither surface hides it)
            gx, gy, gzb, gzt = gl_line(base, top)
            ax, ay, at, ab = [], [], [], []
            for c in range(TILE):
                ax += list(gx)
                ay += list(gy + c * W / 1e3)
                at += list(gzt + 5.0)
                ab += list(gzb - 5.0)
            for z_, nm in ((at, "grounding line (surface)"), (ab, "grounding line (base)")):
                tr.append(go.Scatter3d(x=ax, y=ay, z=z_, mode="lines+markers", line=dict(color="red", width=8),
                                       marker=dict(size=3, color="red"), name=nm,
                                       hovertemplate="GL x %{x:.2f} km at y %{y:.2f} km<extra></extra>"))
        if top is not None and TILE > 1:
            m = np.abs(top["y"] - W) < 1
            o = np.argsort(top["x"][m])
            sx, sz = top["x"][m][o] / 1e3, top["v"][m][o] + 15.0
            qx, qy, qz = [], [], []
            for c in range(1, TILE):
                qx += list(sx) + [None]
                qy += [c * W / 1e3] * len(sx) + [None]
                qz += list(sz) + [None]
            tr.append(go.Scatter3d(x=qx, y=qy, z=qz, mode="lines", line=dict(color="white", width=3, dash="dash"),
                                   name="periodic seam (y = W of one copy = y = 0 of the next)", hoverinfo="skip"))
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
    fig.update_layout(
        title=f"{sys.argv[1] if len(sys.argv) > 1 else ''}: {data[0][0]}",
        scene=dict(xaxis=dict(title="x (km)", range=[x0, x1]), yaxis=dict(title="y (km)"),
                   zaxis=dict(title="z (m)", range=[min(zs) - 20, max(zs) + 20]),
                   aspectmode="manual", aspectratio=asp),
        sliders=[dict(steps=steps, currentvalue=dict(prefix="state: "), pad=dict(t=30))],
        height=850, template="plotly_white", legend=dict(x=0, y=1))
    out = os.path.splitext(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "plot_state_3d")[0] + "_3d.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
