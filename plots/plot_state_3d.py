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
    python plots/plot_state_3d.py STATE.npz --mesh --mesh-dir CASE/footprint   # the mesh's elements on the ice base,
                                                                            # white (without --mesh-dir: Delaunay)
    python plots/plot_state_3d.py STATE.npz --partitions --mesh-dir CASE/footprint   # partition outlines, white
With --switch, buttons above the velocity panel switch its colours between speed, vx, vy and vz.
"mesh on" / "mesh off" (above the temperature panel) shows the element edges on the base in white (drawn at the first
state's base; off by default (no mesh, no buttons): --mesh-toggle for the buttons, --mesh: on from the start; --mesh-dir: the real elements,
else Delaunay triangles of the nodes).
Streamlines of the surface velocity on the surface of both panels (white), one per
state; "streamlines off" / "streamlines on" hides them; --stream-spacing KM (default 5) between their seeds across the
flow; --no-streamlines: none. "play" / "pause" next to the slider runs through the states (--play-ms MS per state,
default 400).
--compare FILE [FILE ...] [--]: a second run (e.g. another time step) in the same scene, beside the first along y,
at the times both have (by frame time), speed coloured the same (--compare-T: temperature); the temperature panel is
off.
Many states make a big page: --every N keeps every N-th state (and the last), --every-yr Y the first frame and then
only frames at multiples of Y yr (a denser start: VTU_FIRST), --dense-until T every frame up to T yr and the thinning
only after it, --years T0 T1 only frames in that time
range. The speed/vx/vy/vz buttons are off (their data was a large share of the page); --switch brings them back.
    python plots/plot_state_3d.py work/<config>/runs/<run>                  # a run with ESS_VTU_EVERY: every step
    python plots/plot_state_3d.py A.npz --diff B.npz                      # colours: A minus B (same mesh)
    python plots/plot_state_3d.py STATE.npz --full                         # the whole sheet, divide to front
    python plots/plot_state_3d.py STATES... --no-switch                    # no vx/vy/vz buttons (a smaller page)
Speed colours max out at 100 m/yr (--vmax V; --vmax-gl: the surface speed on the grounding line; --vmax-all: the largest).
Structured footprints and the unstructured triangle ones (configs/*_tri.toml; Delaunay, needs scipy).
With a Temperature field: a second panel next to it, the same geometry coloured by temperature (surface, base and
side walls; the base shows the bed temperature, where a seam stripe from wrong loads would show).
A treatment folder is read phase by phase (spinup, retreat, advance: <phase>_chunks/points/state_*.npz).
--sheet: instead of the temperature, two panels of the bed seen from above (the base only; THERMAL=elmerstreams runs):
the temperature (surface and base; --sheet-velocity: the velocity instead), the water sheet thickness and the melt
rate (--mask-panel: the cold/temperate mask instead; --melt-max M: its colours
-M .. M m/yr, default its 2nd .. 98th percentile on the grounded temperate bed; red freeze-on, white 0, blue melt). The sheet: linear over its min .. max
in all frames (white 0, blue positive, red negative; --hw-max H:
0 .. H m, thicker saturating; --hw-log: a signed log scale, ticks 1 cm .. 1 km), and the cold/temperate mask (red temperate, blue cold, grey floating); the cold/temperate boundary in magenta
on all three ("temperate on").
Writes <first input>_3d.html and opens it. Needs numpy and plotly.
"""
import base64
import glob
import json
import re
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
SECTOR_A = 0.0           # its angle (radians)
SECTOR = False           # a sector footprint (configs/coarse_sector*.toml: symmetric about the x-axis, so y < 0 on one
                         # half): walls on the angled sides and on the arcs, triangles in (r, angle), no tiling
TILE = 2                 # copies side by side across y (periodicity errors show at the seams); --tile N
VMAX_GL = True           # speed colours up to the surface speed on the grounding line (--vmax-all: the largest)
VMAX = 100.0             # speed colour maximum (m/yr), --vmax V; --vmax-gl: the surface speed on the grounding line
# speed colours: Schoof & Mantelli (2021, Proc. R. Soc. A 477, 20200870) figs 4-7: pale cyan (slow) through blue to
# violet (fast), linear; sampled from their colour bar
SPEED_CS = [[0.0, "#d5fcfe"], [0.0625, "#a4ecfc"], [0.125, "#95ddfb"], [0.1875, "#8bcefa"], [0.25, "#86c0fa"],
            [0.3125, "#81b1f9"], [0.375, "#7ea2f8"], [0.4375, "#7d94f8"], [0.5, "#7d86f7"], [0.5625, "#7e77f7"],
            [0.625, "#8068f2"], [0.6875, "#855bf6"], [0.75, "#894df5"], [0.8125, "#8f40f6"], [0.875, "#9536f6"],
            [0.9375, "#9c2cf6"], [1.0, "#a626f6"]]
SHOW_PARTS = False       # --partitions: the partition outlines in white (needs --mesh-dir with a partitioning)
MESH_DIR = None          # --mesh-dir DIR: the footprint mesh (ElmerGrid directory) whose elements --mesh draws
FS_SSA = 0.0             # km: draw the full-Stokes | SSA boundary this far beyond the grounding line (--fs-ssa 10)
WALLS = False            # walled sides: the distance to the grounding line not wrapped across y (--walls)
SHEET = False            # --sheet: the second panel the water sheet (base only) instead of the temperature
HW_MAX = None            # --hw-max H: a linear colour scale 0 .. H m; default: linear over the data's min .. max
HW_LOG = False           # --hw-log: the signed log scale instead
SHEET_VEL = False        # --sheet-velocity: the first --sheet panel the velocity instead of the temperature
MASK_PANEL = False       # --mask-panel: the third --sheet panel the cold/temperate mask instead of the melt rate
MELT_MAX = None          # --melt-max M: the melt colours -M .. M m/yr (default: 99.5th percentile of |melt|, all frames)
HW_H0 = 0.01             # the signed log scale: sign(h) log10(1 + |h| / HW_H0), from -1e3 to 1e3 m
TEMP = True              # the temperature panel next to the velocity one (if the states have Temperature); --no-temp
MESH_TOGGLE = False      # the "mesh off / mesh on" buttons for the base mesh (white): only with --mesh (or
                         # --mesh-toggle); by default no mesh at all (smaller pages)
SWITCH = False           # the speed/vx/vy/vz buttons (their values for every state go into the page: big); --switch
T_RANGE = (None, None)   # temperature colour range (K); None: all temperatures in the window (--tmin / --tmax)
SHOW = True
COMPARE = []             # --compare FILE [FILE ...] --: a second run's speed beside the first's, at the same times
CMP_FIELD = "speed"      # --compare-T: both runs coloured by temperature instead of speed
PLAY_MS = 400            # ms per state when playing (--play-ms MS)
STREAM = True            # streamlines of the surface velocity on the surface of both panels (--no-streamlines: none)
STREAM_KM = 5.0          # km between the streamlines' seeds across the flow (--stream-spacing KM)


def collect(args):
    """[(label, path)] in order"""
    out = []
    for a in args:
        if a.endswith(".npz") and is_frames(a):   # plots/vtu_frames.py: every output step of one long run
            t = _frames(a)["frame_times"]
            out += [(f"t {v:g} yr", f"{a}#{k}") for k, v in enumerate(t)]
            continue
        if a.endswith(".npz"):
            out.append((os.path.basename(a)[:-4], a))
            continue
        if a.endswith((".pvtu", ".vtu")):        # one step of a run with ESS_VTU_EVERY (base_t*.pvtu)
            out.append((os.path.basename(a).split(".")[0], a))
            continue
        steps = vtu_steps(a)
        if steps:                                # a run directory with ESS_VTU_EVERY: every step (base_t*.pvtu)
            out += [("step %d" % int(re.findall(r"_t(\d+)", os.path.basename(f))[0]), f) for f in steps]
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


_FRAMES = {}


def is_frames(path):
    """a frames file of plots/vtu_frames.py (not a state file)"""
    try:
        with np.load(path, allow_pickle=False) as z:
            return "frame_times" in z.files
    except Exception:
        return False


def _frames(path):
    if path not in _FRAMES:
        with np.load(path, allow_pickle=False) as z:
            _FRAMES[path] = {k: z[k] for k in z.files}
    return _FRAMES[path]


def frame_surfaces(path):
    """top and base of frame k of a frames file ("FILE.npz#k"), as surfaces()"""
    f, k = path.rsplit("#", 1)
    F, k = _frames(f), int(k)

    def side(name):
        x, y = F[f"{name}_x"].astype(float), F[f"{name}_y"].astype(float)
        d = dict(g=np.arange(len(x)), x=x, y=y, v=F[f"{name}_z"][k].astype(float))
        if f"{name}_vx" in F:
            for n in ("vx", "vy", "vz"):
                d[n] = F[f"{name}_{n}"][k].astype(float)
            d["speed"] = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2 + d["vz"] ** 2)
        for n in ("T", "mask", "ct", "hw", "melt", "cfl", "dz"):
            if f"{name}_{n}" in F:
                d[n] = F[f"{name}_{n}"][k].astype(float)
        if f"{name}_bed" in F:
            d["bed"] = F[f"{name}_bed"].astype(float)
        return d
    return side("top"), side("base"), float(F["frame_times"][k])


def vtu_steps(d):
    """the per-step base VTUs of a run directory (ESS_VTU_EVERY: base_t0001.pvtu, ...; serial: base_t0001.vtu)"""
    if not os.path.isdir(d):
        return []
    f = sorted(glob.glob(os.path.join(d, "base_t*.pvtu")))
    return f or sorted(g for g in glob.glob(os.path.join(d, "base_t*.vtu")) if "par" not in os.path.basename(g))


def read_vtu(path):
    """points (n, 3) and point data {lower-case name: (n, components)} of a .vtu (ascii, or raw appended binary with
    UInt32 or UInt64 block headers, as Elmer's VtuOutputSolver writes) or of all pieces of a .pvtu"""
    import xml.etree.ElementTree as ET
    if path.endswith(".pvtu"):
        root = ET.parse(path).getroot()
        srcs = [p.get("Source") for p in root.iter("Piece")]
        parts = [read_vtu(os.path.join(os.path.dirname(path), q)) for q in srcs]
        parts = [q for q in parts if len(q[0])]
        if not parts:
            return np.zeros((0, 3)), {}
        names = set.intersection(*[set(q[1]) for q in parts])
        return (np.concatenate([q[0] for q in parts]),
                {n: np.concatenate([q[1][n] for q in parts]) for n in names})
    raw = open(path, "rb").read()
    k = raw.find(b"<AppendedData")
    head = raw if k < 0 else raw[:k]
    data0 = -1
    if k >= 0:
        data0 = raw.index(b"_", k) + 1
    txt = head.decode("latin-1")
    hdr = np.uint64 if 'header_type="UInt64"' in txt else np.uint32
    order = ">" if 'byte_order="BigEndian"' in txt else "<"
    types = {"Float32": "f4", "Float64": "f8", "Int32": "i4", "Int64": "i8", "UInt8": "u1", "UInt32": "u4",
             "UInt64": "u8", "Int8": "i1"}
    pieces = re.findall(r"<Piece[^>]*NumberOfPoints=\"(\d+)\"", txt)
    npts = int(pieces[0]) if pieces else 0

    def array(tag, body):
        typ = types[re.search(r'type="(\w+)"', tag).group(1)]
        nc = int(re.search(r'NumberOfComponents="(\d+)"', tag).group(1)) if "NumberOfComponents" in tag else 1
        if 'format="appended"' in tag:
            off = int(re.search(r'offset="(\d+)"', tag).group(1))
            hs = np.dtype(hdr).itemsize
            nb = int(np.frombuffer(raw, dtype=order + np.dtype(hdr).str[1:], count=1, offset=data0 + off)[0])
            v = np.frombuffer(raw, dtype=order + typ, count=nb // np.dtype(typ).itemsize, offset=data0 + off + hs)
        else:
            v = np.array(body.split(), dtype=typ)
        return v.astype(float).reshape(-1, nc)
    pts = np.zeros((0, 3))
    m = re.search(r"<Points>\s*(<DataArray[^>]*>)(.*?)</DataArray>|<Points>\s*(<DataArray[^>]*/>)", txt, re.S)
    if m:
        pts = array(m.group(1) or m.group(3), m.group(2) or "")
    pdata = {}
    pd = re.search(r"<PointData[^>]*>(.*?)</PointData>", txt, re.S)
    if pd:
        for mm in re.finditer(r"(<DataArray[^>]*?)(/>|>(.*?)</DataArray>)", pd.group(1), re.S):
            tag = mm.group(1)
            name = re.search(r'Name="([^"]*)"', tag).group(1).lower()
            pdata[name] = array(tag, mm.group(3) or "")
    return pts[:npts] if npts else pts, pdata


def vtu_surfaces(path):
    """top and base of one step from its base_t*.(p)vtu and surface_t*.(p)vtu (ESS_VTU_EVERY), as surfaces()"""
    def surf(p, base):
        X, D = read_vtu(p)
        if not len(X):
            return None
        _, first = np.unique(np.c_[np.round(X[:, 0], 0), np.round(X[:, 1], 0)], axis=0, return_index=True)
        first = np.sort(first)           # nodes shared between partitions once
        X = X[first]
        D = {k: v[first] for k, v in D.items()}
        get = lambda *names: next((D[n] for n in names if n in D), None)
        u = get("velocity")
        d = dict(g=np.arange(len(X)), x=X[:, 0], y=X[:, 1], v=X[:, 2])
        if u is not None:
            d["vx"], d["vy"], d["vz"] = (u[:, c] for c in range(3))
            d["speed"] = np.hypot(u[:, 0], u[:, 1]) if u.shape[1] >= 2 else np.abs(u[:, 0])
            d["speed"] = np.sqrt(np.sum(u[:, :3] ** 2, axis=1))
        T = get("temperature")
        if T is not None:
            d["T"] = T[:, 0]
        if base:
            gm = get("groundedmask")
            if gm is not None:
                d["mask"] = gm[:, 0]
            b = get("bedrock")
            if b is not None:
                d["bed"] = b[:, 0]
        return d
    base = surf(path, True)
    top = surf(path.replace("base_t", "surface_t"), False)
    step = re.findall(r"_t(\d+)", os.path.basename(path))
    return top, base, float(step[0]) if step else np.nan


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
    if path.endswith((".pvtu", ".vtu")):
        return vtu_surfaces(path)
    if "#" in os.path.basename(path):
        return frame_surfaces(path)
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
    comps = [dict(zip(r[0].tolist(), np.asarray(r[1], float).tolist())) if r is not None else {} for r in u]

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
        ctr = _rec(s, "ColdTempMask")
        if ctr is not None:
            cm = dict(zip(ctr[0].tolist(), ctr[1].tolist()))
            base["ct"] = np.array([cm.get(g, np.nan) for g in base["g"].tolist()])
        base["speed"] = np.array([sp.get(g, np.nan) for g in base["g"].tolist()])
    if top is not None:
        top["speed"] = np.array([sp.get(g, np.nan) for g in top["g"].tolist()])
    for d in (top, base):                          # the velocity components (the colour switch)
        if d is not None:
            for n, c in zip(("vx", "vy", "vz"), comps):
                d[n] = np.array([c.get(g, np.nan) for g in d["g"].tolist()])
    for d in (top, base):
        if d is not None and Td is not None:
            d["T"] = np.array([Td.get(g, np.nan) for g in d["g"].tolist()])
    return top, base, float(s.meta.get("time", np.nan))


def structured(x, y):
    """a structured footprint (rows of equal y, the same x positions in every row), not unstructured triangles"""
    return len(np.unique(np.round(x, 1))) * len(np.unique(np.round(y, 1))) == len(x)


_TRI_CACHE = {}


def triangles(x, y):
    """triangles of the footprint (cached: every state and panel of a plot has the same footprint)"""
    # keyed on the node ORDER too: a state file (nodes by global id) and a frames file (nodes sorted by x, y) of the same
    # mesh have the same nodes in another order, and the other's triangles connect the wrong nodes (spikes everywhere)
    import hashlib
    key = hashlib.sha1(np.ascontiguousarray(np.round(np.asarray(x, float), 1)).tobytes()
                       + np.ascontiguousarray(np.round(np.asarray(y, float), 1)).tobytes()).hexdigest()
    if key not in _TRI_CACHE:
        _TRI_CACHE[key] = _triangles(x, y)
    return _TRI_CACHE[key]


def sector_angle(x, y):
    """a sector footprint's angle (radians): the span of the nodes on its outer arc (whose two corners are on the
    straight-side line also on a cut sector with jagged sides, ess/case.py sector_geo cut_seed)"""
    r, ph = np.hypot(x, y), np.arctan2(y, x)
    m = r > r.max() - 1.0
    return float(ph[m].max() - ph[m].min()) if m.sum() > 1 else float(np.ptp(ph))


_SIDES_CACHE = {}


def sector_sides(x, y, A=None):
    """the two periodic sides of a sector footprint: (side 1, side 2) node indices, each ordered by radius. Side 1:
    the nodes whose image rotated by the sector angle is a node too (side 2): straight or jagged (a cut sector)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    key = (len(x), float(x.sum()), float(y.sum()))
    if key in _SIDES_CACHE:
        return _SIDES_CACHE[key]
    A = sector_angle(x, y) if A is None else A
    from scipy.spatial import cKDTree
    d, j = cKDTree(np.c_[x, y]).query(np.c_[np.cos(A) * x - np.sin(A) * y, np.sin(A) * x + np.cos(A) * y])
    i1 = np.where(d < 1.0)[0]
    i2 = j[i1]
    o = np.argsort(np.hypot(x[i1], y[i1]))
    _SIDES_CACHE[key] = (i1[o], i2[o])
    return _SIDES_CACHE[key]


def _triangles(x, y):
    """triangles of the footprint: of the rows and columns of a structured one, else Delaunay (the unstructured
    triangle meshes: their footprint is a rectangle, so Delaunay covers it like the mesh)"""
    if not structured(x, y):
        from scipy.spatial import Delaunay
        if np.min(y) < -1.0:      # a sector: Delaunay in (r, arc at the mean radius), where it is a rectangle (in x, y
            r = np.hypot(x, y)    # its inner arc is concave, and Delaunay would bridge it)
            t = Delaunay(np.c_[r, np.arctan2(y, x) * np.mean(r)] / 1e3).simplices
        else:
            t = Delaunay(np.c_[x, y] / 1e3).simplices
        # no slivers: on a boundary that is straight only to round-off (the angled sides of a sector: r cos, r sin of
        # the same angle) Delaunay strings nearly collinear boundary nodes, hundreds of km apart, into flat triangles,
        # and every line drawn through them (grounding line, masks) runs along the side
        P_ = np.c_[x, y]
        a, b, c = P_[t[:, 0]], P_[t[:, 1]], P_[t[:, 2]]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        emax = np.max(np.c_[np.hypot(*(b - a).T), np.hypot(*(c - b).T), np.hypot(*(a - c).T)], axis=1)
        ok = area > 1e-3 * emax ** 2
        if (~ok).any():
            print(f"  footprint triangles: {int((~ok).sum())} slivers dropped (of {len(t)})")
        t = t[ok]
        if np.min(y) < -1.0:      # a sector: no triangle outside it (Delaunay fills the notches of a jagged side)
            A = sector_angle(x, y)
            i1, _ = sector_sides(x, y, A)
            if len(i1) > 1:
                r1, p1 = np.hypot(x[i1], y[i1]), np.arctan2(y[i1], x[i1])
                cx, cy = P_[t].mean(1).T
                rc, pc = np.hypot(cx, cy), np.arctan2(cy, cx)
                rel = pc - np.interp(rc, r1, p1)
                ins = (rel > -1e-9) & (rel < A + 1e-9)
                if (~ins).any():
                    print(f"  footprint triangles: {int((~ins).sum())} outside the sector dropped")
                t = t[ins]
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


def mesh_triangles(mdir, x, y):
    """the triangles of the footprint mesh in mdir (an ElmerGrid mesh directory, its mesh.* or its partitioning's
    part.*), as indices into the nodes (x, y); triangles with a node outside them (the x window) dropped"""
    t, _ = mesh_elements(mdir, x, y)
    if not len(t):
        print(f"  (none of its nodes match the state's: not this state's mesh? Delaunay instead)")
        return triangles(x, y)
    return t[:, 0], t[:, 1], t[:, 2]


_MESH = {}
_SAID = set()


def once(msg):
    """print a message only the first time (the panels are built several times: page, slider frames, buttons)"""
    if msg not in _SAID:
        _SAID.add(msg)
        print(msg)


def mesh_elements(mdir, x, y):
    """the triangles of the mesh in mdir as node indices into (x, y) (triangles with a node outside dropped), and the
    partition of each (0 for a serial mesh). Read once per directory."""
    if mdir not in _MESH:
        if os.path.exists(os.path.join(mdir, "mesh.nodes")):
            nf = [os.path.join(mdir, "mesh.nodes")]
        else:
            nf = sorted(glob.glob(os.path.join(sorted(glob.glob(os.path.join(mdir, "partitioning.*")))[0],
                                               "part.*.nodes")))
        xy, els, part = {}, {}, {}
        for f in nf:
            pn = int(os.path.basename(f).split(".")[1]) if os.path.basename(f).startswith("part.") else 0
            for line in open(f):
                s = line.split()
                xy[int(s[0])] = (round(float(s[2]), 0), round(float(s[3]), 0))
            for line in open(f.replace(".nodes", ".elements")):
                s = line.split()
                if s[2] == "303":
                    els[int(s[0])] = [int(v) for v in s[3:6]]
                    part[int(s[0])] = pn
        _MESH[mdir] = (xy, els, part)
    xy, els, part = _MESH[mdir]
    here = {(round(a, 0), round(b, 0)): n for n, (a, b) in enumerate(zip(x, y))}
    t, p = [], []
    for e, nodes in els.items():
        q = [here.get(xy[v], -1) for v in nodes]
        if min(q) >= 0:
            t.append(q)
            p.append(part[e])
    once(f"  mesh: {len(t)} of the {len(els)} triangles of {mdir} in the window")
    return np.array(t, int).reshape(-1, 3), np.array(p, int)


def partition_edges(t, p):
    """the edges between triangles of different partitions (node index pairs)"""
    e = np.r_[t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]
    pe = np.r_[p, p, p]
    e = np.sort(e, axis=1)
    o = np.lexsort((e[:, 1], e[:, 0]))
    e, pe = e[o], pe[o]
    same = np.all(e[1:] == e[:-1], axis=1) & (pe[1:] != pe[:-1])
    return e[1:][same]


def contour(x, y, f, level, zs):
    """the level line of the nodal field f on the footprint triangles, as segments (x, y in km, then each of the
    nodal fields zs interpolated onto it), segments separated by None (plotly lines). Vectorised over the triangles."""
    i, j, k = triangles(x, y)
    T = np.c_[i, j, k]
    A, B = T, T[:, [1, 2, 0]]                     # edges (t0, t1), (t1, t2), (t2, t0)
    f = np.asarray(f, float)
    fa, fb = f[A] - level, f[B] - level
    with np.errstate(invalid="ignore"):
        cr = ((fa < 0) != (fb < 0)) & np.isfinite(fa) & np.isfinite(fb)
    keep = cr.sum(1) == 2
    A, B, fa, fb, cr = A[keep], B[keep], fa[keep], fb[keep], cr[keep]
    r = np.arange(len(A))
    e1 = np.argmax(cr, 1)
    cr2 = cr.copy()
    cr2[r, e1] = False
    e2 = np.argmax(cr2, 1)
    vals = [np.asarray(x, float), np.asarray(y, float)] + [np.asarray(z, float) for z in zs]

    def at(e):
        a, b = A[r, e], B[r, e]
        w = fa[r, e] / (fa[r, e] - fb[r, e])
        return [v[a] + w * (v[b] - v[a]) for v in vals]
    p, q = at(e1), at(e2)
    out = []
    for c in range(len(vals)):
        sc = 1e3 if c < 2 else 1.0
        arr = np.empty(3 * len(r), dtype=object)
        arr[0::3] = (p[c] / sc).tolist()
        arr[1::3] = (q[c] / sc).tolist()
        arr[2::3] = None
        out.append(arr.tolist())
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
    if periodic and base["y"].min() < -1.0:      # a sector: the images rotated by +- the sector angle
        A = sector_angle(base["x"], base["y"])
        rot = lambda a: np.c_[np.cos(a) * pts[:, 0] - np.sin(a) * pts[:, 1], np.sin(a) * pts[:, 0] + np.cos(a) * pts[:, 1]]
        pts = np.r_[pts, rot(A), rot(-A)]
    elif periodic:
        pts = np.r_[pts, pts + [0, W], pts - [0, W]]
    q = np.c_[base["x"][fl], base["y"][fl]] / 1e3
    try:
        from scipy.spatial import cKDTree
        d[fl] = cKDTree(pts).query(q)[0]
    except ImportError:
        for c in np.array_split(np.arange(len(fl)), max(1, len(fl) // 2000)):
            d[fl[c]] = np.sqrt(((q[c, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(1)
    return d


def tile_xy(xs, ys, c, W):
    """copy c of the points (km; None for gaps): the strip shifted by c W in y, a sector rotated by c times its angle"""
    if c == 0:
        return list(xs), list(ys)
    if SECTOR:
        ca, sa = np.cos(c * SECTOR_A), np.sin(c * SECTOR_A)
        X = [None if a is None or b is None else ca * a - sa * b for a, b in zip(xs, ys)]
        Y = [None if a is None or b is None else sa * a + ca * b for a, b in zip(xs, ys)]
        return X, Y
    return list(xs), [None if b is None else b + c * W / 1e3 for b in ys]


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


_STREAM_CACHE = {}


_LOC_CACHE = {}


def _point_locator(P, S, key):
    """locate points in the triangles S of the points P: (triangle index or -1, barycentric weights). Without qhull's
    find_simplex (it crashed with "Illegal instruction" on pilsen, as scipy's interpolators that use it): the nearest
    nodes (cKDTree), then every triangle around them tested by its barycentric coordinates"""
    if key in _LOC_CACHE:
        return _LOC_CACHE[key]
    from scipy.spatial import cKDTree
    P = np.asarray(P, float)
    tree = cKDTree(P)
    # node -> its triangles, padded (-1)
    cnt = np.bincount(S.ravel(), minlength=len(P))
    mx = int(cnt.max())
    inc = -np.ones((len(P), mx), int)
    order = np.argsort(S.ravel(), kind="stable")
    tri_of = order // 3
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    pos = np.arange(len(order)) - np.repeat(start, cnt)
    inc[np.sort(S.ravel())[np.arange(len(order))], pos] = tri_of
    a, b, c = P[S[:, 0]], P[S[:, 1]], P[S[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    def locate(Q, kn=4):
        Q = np.asarray(Q, float)
        _, nn = tree.query(Q, k=min(kn, len(P)))
        cand = inc[nn].reshape(len(Q), -1)                      # (n, 4 mx) triangles around the 4 nearest nodes
        valid = cand >= 0
        cs = np.where(valid, cand, 0)
        A_, B_, C_ = a[cs], b[cs], c[cs]
        qx, qy = Q[:, None, 0], Q[:, None, 1]
        d = det[cs]
        l1 = ((B_[..., 0] - qx) * (C_[..., 1] - qy) - (B_[..., 1] - qy) * (C_[..., 0] - qx)) / d
        l2 = ((C_[..., 0] - qx) * (A_[..., 1] - qy) - (C_[..., 1] - qy) * (A_[..., 0] - qx)) / d
        l3 = 1.0 - l1 - l2
        inside = valid & (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        hit = inside.any(1)
        j = np.argmax(inside, 1)
        r = np.arange(len(Q))
        k = np.where(hit, cand[r, j], -1)
        w = np.c_[l1[r, j], l2[r, j], l3[r, j]]
        miss = ~hit
        if kn < 64 and miss.any():                             # long triangles (at the edges): more nodes around
            k2, w2 = locate(Q[miss], 4 * kn)
            k[miss], w[miss] = k2, w2
        return k, w
    _LOC_CACHE[key] = locate
    return locate


def streamlines(top, spacing_km, width_m, periodic=True):
    """streamlines of the surface velocity (vx, vy): seeds every spacing_km across the strip, 20 km downstream of the
    window's upstream end, followed along the flow direction (RK2, 0.5 km steps, a point kept every 2 km) to the front,
    a stagnant point or the edge; periodic in y unless walls (a line leaving one side carries on from the other, with a
    gap). Returns x, y (km) and z (m, just above the surface) with None between the pieces"""
    key = id(top)
    if key in _STREAM_CACHE:
        return _STREAM_CACHE[key]
    from scipy.spatial import Delaunay           # (no scipy.interpolate: it crashed with "Illegal instruction"
    import hashlib                               # on pilsen; the barycentric interpolation is done here instead)
    x, y = np.asarray(top["x"], float), np.asarray(top["y"], float)
    tk = "stream" + hashlib.sha1(np.round(x, 1).tobytes() + np.round(y, 1).tobytes()).hexdigest()
    if tk not in _TRI_CACHE:
        _TRI_CACHE[tk] = Delaunay(np.c_[x, y])
    tri_, vals_ = _TRI_CACHE[tk], np.c_[top["vx"], top["vy"], top["v"]]

    loc_ = _point_locator(tri_.points, tri_.simplices, tk)

    def f(Q):                                    # linear on the Delaunay triangles, NaN outside
        k, w = loc_(Q)
        out = np.full((len(Q), 3), np.nan)
        ok = k >= 0
        if ok.any():
            out[ok] = np.einsum("ni,nij->nj", w[ok], vals_[tri_.simplices[k[ok]]])
        return out
    x0, x1, y0 = x.min(), x.max(), y.min()
    W = width_m if width_m > 0 else float(np.ptp(y))
    n = max(1, int(round(W / (spacing_km * 1e3))))
    P = np.c_[np.full(n, min(x0 + 20e3, 0.5 * (x0 + x1))), y0 + (np.arange(n) + 0.5) * W / n]
    ds, keep = 500.0, 4
    lines = [[tuple(p)] for p in P]
    alive = np.ones(n, bool)

    def wrap(Q):
        if periodic:
            Q[:, 1] = y0 + np.mod(Q[:, 1] - y0, W)
        return Q

    def direction(Q):
        u = f(Q)
        sp = np.hypot(u[:, 0], u[:, 1])
        ok = np.isfinite(sp) & (sp > 1e-3)
        d = np.zeros((len(Q), 2))
        d[ok] = u[ok, :2] / sp[ok, None]
        return d, ok

    for it in range(int(3 * (x1 - x0) / ds)):
        if not alive.any():
            break
        a = np.where(alive)[0]
        d1, ok1 = direction(P[a])
        M = wrap(P[a] + 0.5 * ds * d1)
        d2, ok2 = direction(M)
        Q = P[a] + ds * d2
        jump = periodic & ((Q[:, 1] < y0) | (Q[:, 1] > y0 + W))
        Q = wrap(Q)
        stop = ~(ok1 & ok2) | (Q[:, 0] >= x1) | (Q[:, 0] <= x0) | ((not periodic) & ((Q[:, 1] < y0) | (Q[:, 1] > y0 + W)))
        for q, i in enumerate(a):
            if stop[q]:
                alive[i] = False
                continue
            if jump[q]:
                lines[i].append(None)
            if jump[q] or it % keep == keep - 1:
                lines[i].append((Q[q, 0], Q[q, 1]))
        P[a] = np.where(stop[:, None], P[a], Q)
    # each piece thinned to the points that matter (Ramer-Douglas-Peucker: within 100 m across and 4 m in height of
    # the full line; the long straight stretches become a few points), z 15 m above the surface; float32 with NaN
    # between the pieces (plotly >= 6 writes these as binary: the page stays small with a line set per state)
    out = []
    for ln in lines:
        pieces, cur = [], []
        for p in ln + [None]:
            if p is None:
                if len(cur) > 1:
                    pieces.append(np.array(cur))
                cur = []
            else:
                cur.append(p)
        for pc in pieces:
            z = f(pc)[:, 2]
            ok = np.isfinite(z)
            pc, z = pc[ok], z[ok]
            if len(pc) < 2:
                continue
            k = _rdp(np.c_[pc / 100.0, z / 4.0], 1.0)
            out.append(np.c_[pc[k] / 1e3, z[k] + 15.0])
            out.append(np.full((1, 3), np.nan))
    a = np.concatenate(out).astype(np.float32) if out else np.zeros((0, 3), np.float32)
    _STREAM_CACHE[key] = (a[:, 0], a[:, 1], a[:, 2])
    return _STREAM_CACHE[key]


_SSTREAM_CACHE = {}


def streamlines_sector(top, spacing_km):
    """streamlines on a sector: traced by streamlines() in the unrolled coordinates X = distance along the flow from the
    upstream arc (r, or r_max - r when the ice flows inwards: an inverted sector), Y = arc length at the mean radius
    (periodic across the sides by the arc's width), the velocity turned and scaled to match; mapped back to x, y"""
    key = id(top)
    if key in _SSTREAM_CACHE:
        return _SSTREAM_CACHE[key]
    x, y = np.asarray(top["x"], float), np.asarray(top["y"], float)
    r, ph = np.hypot(x, y), np.arctan2(y, x)
    vr = (top["vx"] * x + top["vy"] * y) / r
    vt = (-top["vx"] * y + top["vy"] * x) / r
    inward = np.nanmean(vr) < 0
    A = sector_angle(x, y)
    R, p0, rmax = float(np.mean(r)), -0.5 * A, float(r.max())
    X = (rmax - r) if inward else r
    m = dict(x=X, y=(ph - p0) * R, vx=(-vr if inward else vr), vy=vt * R / r, v=np.asarray(top["v"], float))
    sx, sy, sz = streamlines(m, spacing_km, A * R, periodic=True)
    rr = (rmax / 1e3 - sx) if inward else sx
    pp = sy * 1e3 / R + p0
    out = ((rr * np.cos(pp)).astype(np.float32), (rr * np.sin(pp)).astype(np.float32), sz)
    _SSTREAM_CACHE[key] = out
    return out


def _rdp(P, eps):
    """the indices of P (n x d) kept by Ramer-Douglas-Peucker with tolerance eps (iterative)"""
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L = float(ab @ ab)
        Q = P[i + 1:j] - a
        if L > 0:
            t = np.clip(Q @ ab / L, 0, 1)
            d = np.linalg.norm(Q - t[:, None] * ab, axis=1)
        else:
            d = np.linalg.norm(Q, axis=1)
        m = int(np.argmax(d))
        if d[m] > eps:
            keep[i + 1 + m] = True
            stack += [(i, i + 1 + m), (i + 1 + m, j)]
    return np.where(keep)[0]


def main():
    import plotly.graph_objects as go
    global SHEET, HW_MAX, HW_LOG, MASK_PANEL, MELT_MAX, SHEET_VEL
    global X_WINDOW, TILE, TEMP, T_RANGE, SHOW_MESH, SHOW_BED, FS_SSA, WALLS, VMAX, VMAX_GL, MESH_DIR, SHOW_PARTS, SWITCH
    global MESH_TOGGLE, STREAM, STREAM_KM, PLAY_MS, COMPARE, CMP_FIELD, SECTOR, SECTOR_A
    args = sys.argv[1:]
    if "--vmax-all" in args:
        VMAX_GL, VMAX = False, None
        args.remove("--vmax-all")
    if "--vmax-gl" in args:
        VMAX = None
        args.remove("--vmax-gl")
    if "--vmax" in args:
        k = args.index("--vmax")
        VMAX = float(args[k + 1])
        args = args[:k] + args[k + 2:]
    DIFF = None
    if "--diff" in args:                          # --diff OTHER.npz: colours = this state minus OTHER (same mesh)
        k = args.index("--diff")
        DIFF = args[k + 1]
        args = args[:k] + args[k + 2:]
    global SHOW
    if "--no-show" in args:                       # only write the page (e.g. to view it from another machine)
        SHOW = False
        args.remove("--no-show")
    if "--full" in args:                          # the whole ice sheet, divide to front
        X_WINDOW = (-1.0, 1e9)
        args.remove("--full")
    if "--partitions" in args:
        SHOW_PARTS = True
        args.remove("--partitions")
    if "--mesh-dir" in args:
        k = args.index("--mesh-dir")
        MESH_DIR = args[k + 1]
        args = args[:k] + args[k + 2:]
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
        SHOW_MESH = MESH_TOGGLE = True
        args.remove("--mesh")
    if "--mesh-toggle" in args:
        MESH_TOGGLE = True
        args.remove("--mesh-toggle")
    if "--no-temp" in args:
        TEMP = False
        args.remove("--no-temp")
    if "--sheet" in args:
        SHEET = True
        args.remove("--sheet")
    if "--sheet-velocity" in args:
        SHEET_VEL = True
        args.remove("--sheet-velocity")
    if "--mask-panel" in args:
        MASK_PANEL = True
        args.remove("--mask-panel")
    if "--melt-max" in args:
        k = args.index("--melt-max")
        MELT_MAX = float(args[k + 1])
        del args[k:k + 2]
    if "--hw-log" in args:
        HW_LOG = True
        args.remove("--hw-log")
    if "--hw-max" in args:
        k = args.index("--hw-max")
        HW_MAX = float(args[k + 1])
        del args[k:k + 2]
    if "--switch" in args:
        SWITCH = True
        args.remove("--switch")
    if "--no-switch" in args:
        SWITCH = False
        args.remove("--no-switch")
    if "--compare-T" in args:
        CMP_FIELD = "T"
        args.remove("--compare-T")
    if "--compare" in args:                       # --compare FILE [FILE ...] [--]: the files up to "--" or the next option
        k = args.index("--compare")
        j = k + 1
        while j < len(args) and not args[j].startswith("--"):
            COMPARE.append(args[j]); j += 1
        args = args[:k] + args[j + (1 if j < len(args) and args[j] == "--" else 0):]
    if "--play-ms" in args:                       # --play-ms MS: the time per state of the play button
        k = args.index("--play-ms")
        PLAY_MS = int(float(args[k + 1]))
        args = args[:k] + args[k + 2:]
    if "--no-streamlines" in args:
        STREAM = False
        args.remove("--no-streamlines")
    if "--stream-spacing" in args:               # --stream-spacing KM: the streamlines' seed spacing across the flow
        k = args.index("--stream-spacing")
        STREAM_KM = float(args[k + 1])
        args = args[:k] + args[k + 2:]
    if "--no-mesh-toggle" in args:
        MESH_TOGGLE = False
        args.remove("--no-mesh-toggle")
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
    every, tsel, every_yr, dense = 1, None, None, None
    if "--dense-until" in args:                   # --dense-until T: every frame up to T yr, the thinning (--every,
        k = args.index("--dense-until")           # --every-yr) only after it
        dense = float(args[k + 1])
        args = args[:k] + args[k + 2:]
    if "--every-yr" in args:                      # --every-yr Y: the first frame, then only frames at multiples of Y yr
        k = args.index("--every-yr")              # (and the last): e.g. 500 with a denser start (VTU_FIRST)
        every_yr = float(args[k + 1])
        args = args[:k] + args[k + 2:]
    if "--every" in args:                         # --every N: every N-th state (the last one always kept)
        k = args.index("--every")
        every = max(1, int(args[k + 1]))
        args = args[:k] + args[k + 2:]
    if "--years" in args:                         # --years T0 T1: only the frames with T0 <= t <= T1 (frames files)
        k = args.index("--years")
        tsel = (float(args[k + 1]), float(args[k + 2]))
        args = args[:k] + args[k + 2 + 1:]
    if SHOW_PARTS and not MESH_DIR:
        raise SystemExit("--partitions needs --mesh-dir CASE/footprint (with its partitioning.N)")
    sys.argv = [sys.argv[0]] + args
    items = collect(args or ["work/test_periodic_gl/fix"])
    if tsel:
        tv = lambda lab: float(lab[2:-3]) if lab.startswith("t ") and lab.endswith(" yr") else None
        items = [it for it in items if tv(it[0]) is None or tsel[0] - 1e-6 <= tv(it[0]) <= tsel[1] + 1e-6]
    if COMPARE:                                   # only the times the compared run has (before loading anything)
        tv_ = lambda lab: float(lab[2:-3].replace(",", "")) if lab.startswith("t ") and lab.endswith(" yr") else None
        ct = {round(tv_(l), 0) for l, _ in collect(COMPARE) if tv_(l) is not None}
        items = [it for it in items if tv_(it[0]) is None or round(tv_(it[0]), 0) in ct]
    tvy = lambda lab: float(lab[2:-3].replace(",", "")) if lab.startswith("t ") and lab.endswith(" yr") else None
    early = lambda it: dense is not None and tvy(it[0]) is not None and tvy(it[0]) <= dense + 1e-9
    if every_yr and len(items) > 1:
        items = [it for n, it in enumerate(items)
                 if n in (0, len(items) - 1) or tvy(it[0]) is None or early(it)
                 or abs(tvy(it[0]) / every_yr - round(tvy(it[0]) / every_yr)) < 1e-6]
        print(f"  frames: the first, then every {every_yr:g} yr" + (f" (all up to {dense:g} yr)" if dense else ""))
    if every > 1 and len(items) > 1:
        ne = sum(1 for it in items if early(it))
        items = [it for n, it in enumerate(items) if early(it) or (n - ne) % every == 0 or n == len(items) - 1]
        if dense:
            print(f"  frames: all up to {dense:g} yr, then every {every}-th")
    print(f"{len(items)} states" + (f" (every {every}-th)" if every > 1 else "")
          + (f" (years {tsel[0]:g} .. {tsel[1]:g})" if tsel else ""))
    data = []
    for label, f in items:
        top, base, t = surfaces(f)
        if DIFF:                                  # the fields minus the other state's, node by node (global ids)
            otop, obase, _ = surfaces(DIFF)
            for d, o in ((top, otop), (base, obase)):
                if d is None or o is None:
                    continue
                pos = {g: i for i, g in enumerate(o["g"].tolist())}
                idx = np.array([pos.get(g, -1) for g in d["g"].tolist()])
                for key in ("speed", "vx", "vy", "vz", "T"):
                    if key in d and key in o:
                        d[key] = np.where(idx >= 0, d[key] - o[key][np.maximum(idx, 0)], np.nan)
            label = f"{label} minus {os.path.basename(DIFF)[:-4]}"
        data.append((label, top, base, t))
        if base is None or "mask" not in base:
            print(f"  {label}: no GroundedMask / Zb in {f}")
        else:
            gx = gl_x(base)
            per = (f"per row {np.round(gx, 2)}" if len(gx) <= 6 else
                   f"min {np.min(gx):.2f} mean {np.mean(gx):.2f} max {np.max(gx):.2f} over {len(gx)} rows")
            print(f"  {label}: grounding line {per} km; mask values {np.unique(base['mask'])}")
    # window around the grounding line
    SECTOR = any(tp is not None and float(np.min(tp["y"])) < -1.0 for _, tp, _, _ in data)
    if SECTOR:
        tp0 = next(tp for _, tp, _, _ in data if tp is not None)
        SECTOR_A = sector_angle(tp0["x"], tp0["y"])
        if TILE > 1:
            print(f"sector footprint: {TILE} copies, each the last one rotated by the sector angle "
                  f"({np.degrees(SECTOR_A):.3f} deg): the seams show whether the sides are periodic in (r, angle)")
        if not X_WINDOW:
            X_WINDOW = (-1.0, 1e9)
    if X_WINDOW:
        x0, x1 = X_WINDOW
    else:
        gls = [gl_x(b) for _, _, b, _ in data if b is not None and "mask" in b]
        gls = [g for g in gls if len(g)]
        xmax = max(float(np.max(top["x"])) for _, top, _, _ in data if top is not None) / 1e3
        x0 = (min(float(np.min(g)) for g in gls if len(g)) - 60.0) if gls else 0.0
        x1 = xmax
    xall = np.concatenate([np.hypot(tp["x"], tp["y"]) if SECTOR else tp["x"] for _, tp, _, _ in data
                           if tp is not None]) / 1e3
    x0, x1 = max(x0, float(xall.min())), min(x1, float(xall.max()))
    print(f"x window {x0:.0f} .. {x1:.0f} km (domain {xall.min():.0f} .. {xall.max():.0f} km; --x XMIN XMAX to change)")

    def crop(d):
        if d is None:
            return None
        xr = np.hypot(d["x"], d["y"]) if SECTOR else d["x"]
        m = (xr / 1e3 >= x0) & (xr / 1e3 <= x1)
        return {k: (v[m] if isinstance(v, np.ndarray) and v.shape == d["x"].shape else v) for k, v in d.items()}
    data = [(l, crop(tp), crop(b), t) for l, tp, b, t in data]
    if COMPARE:                                   # the second run at the same times, beside the first (speed both)
        other = {}
        for _, f in collect(COMPARE):
            tp2, b2, t2 = surfaces(f)
            if t2 is not None:
                other[round(float(t2), 0)] = (crop(tp2), crop(b2))
        kept = []
        for l, tp, b, t in data:
            o = other.get(round(float(t), 0)) if t is not None else None
            if o is not None and tp is not None:
                tp["_cmp"] = o
                kept.append((l, tp, b, t))
        print(f"compare: {len(kept)} of {len(data)} states at times also in {' '.join(COMPARE)}")
        if not kept:
            raise SystemExit("--compare: no common times")
        data = kept
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
    sheet = SHEET and not COMPARE and all(b is not None and "hw" in b for _, _, b, _ in data)
    if SHEET and not sheet:
        print("no Water Sheet Thickness in the frames (a THERMAL=elmerstreams run, packed by plots/vtu_frames.py): "
              "no sheet panel")
    temp = TEMP and not sheet and not COMPARE and all(d is not None and "T" in d for _, tp, b, _ in data for d in (tp, b))
    two = temp or sheet                           # two panels side by side (velocity | temperature or water sheet)
    # the third panel: the melt rate (default), or the cold/temperate mask (--mask-panel, or no melt in the frames)
    p3 = ("melt" if not MASK_PANEL and all("melt" in b for _, _, b, _ in data if b is not None) else
          "ctm" if all("ct" in b and "mask" in b for _, _, b, _ in data if b is not None) else None)
    three = sheet and p3 is not None              # velocity | water sheet | melt rate (or mask)
    ZFLAT = float(min(np.nanmin(b["v"]) for _, _, b, _ in data if b is not None)) - 10.0   # the flat maps' height
    if three and p3 == "ctm":
        for _, _, b, _ in data:                   # the bed's state: 1 temperate, -1 cold (grounded), 0 floating
            b["ctm"] = np.where(b["mask"] < -0.5, 0.0, np.where(np.nan_to_num(b["ct"], nan=-1.0) > 0, 1.0, -1.0))
    if TEMP and not temp:
        print("no Temperature in the states (slimmed before it was kept? re-run plots/slim_states.py): velocity only")
    # --sheet: the first panel the temperature (surface and base), not the velocity (--sheet-velocity to keep it)
    sheetT = sheet and not SHEET_VEL and all(d is not None and "T" in d for _, tp, b, _ in data for d in (tp, b))
    if temp or sheetT or (COMPARE and CMP_FIELD == "T"):
        Tv = np.concatenate([d["T"] for _, tp, b, _ in data for d in (tp, b) if d is not None and "T" in d] +
                            [d["T"] for _, tp, _, _ in data for d in tp.get("_cmp", ()) if d is not None and "T" in d])
        tmin = T_RANGE[0] if T_RANGE[0] is not None else float(np.nanmin(Tv))
        tmax = T_RANGE[1] if T_RANGE[1] is not None else float(np.nanmax(Tv))
        print(f"temperature colours {tmin:.2f} .. {tmax:.2f} K: all temperatures (--tmin / --tmax to narrow)")
    # the panels: (field, colour range, colour scale, colour bar title, colour bar x, scene)
    panels = [("speed", 0.0, smax, SPEED_CS, "speed (m/yr)", 0.30 if three else 0.47 if two else 1.0, "scene")]
    if sheetT:
        panels[0] = ("T", tmin, tmax, "RdBu_r", "temperature (K)", panels[0][5], "scene")
    if DIFF:                                      # differences: vz first, symmetric red/blue
        v = np.abs(np.concatenate([d["vz"] for _, tp, b, _ in data for d in (tp, b) if d is not None]))
        m = max(float(np.nanpercentile(v, 99.5)) if np.isfinite(v).any() else 1.0, 1e-9)
        panels[0] = ("vz", -m, m, "RdBu_r", "difference in vz (m/yr)", panels[0][5], "scene")
        print(f"difference mode: vz colours -{m:.3g} .. {m:.3g} m/yr (99.5th percentile of |difference|), "
              f"largest {np.nanmax(v):.3g} m/yr")
    if temp:
        panels.append(("T", tmin, tmax, "RdBu_r", "temperature (K)", 1.0, "scene2"))
    if sheet:
        hv = np.concatenate([b["hw"] for _, _, b, _ in data])
        if HW_MAX:
            print(f"water sheet colours 0 .. {HW_MAX:g} m (--hw-max); all frames: {np.nanmin(hv):.3g} .. {np.nanmax(hv):.3g} m")
            panels.append(("hw", 0.0, HW_MAX, "Blues", "water sheet (m)", 0.645 if three else 1.0, "scene2"))
        elif not HW_LOG:
            # linear over the data (all frames): white at 0, blue above, red below (if any)
            lo_, hi_ = float(np.nanmin(hv)), float(np.nanmax(hv))
            hi_ = hi_ if hi_ > lo_ else lo_ + 1e-6
            if lo_ < 0 < hi_:
                z0 = -lo_ / (hi_ - lo_)
                cs_ = [[0.0, "#b2182b"], [z0, "#ffffff"], [1.0, "#08306b"]]
            else:
                cs_ = "Blues"
            print(f"water sheet colours: linear {lo_:.3g} .. {hi_:.3g} m, the data's range over all frames "
                  f"(white 0, blue positive, red negative; --hw-max H: 0 .. H; --hw-log: signed log)")
            panels.append(("hw", lo_, hi_, cs_, "water sheet (m)", 0.645 if three else 1.0, "scene2"))
        else:
            # signed log: blue positive, red negative, white 0; every thickness from 1 cm to 1 km distinguishable
            for _, _, b, _ in data:
                b["hwl"] = np.sign(b["hw"]) * np.log10(1.0 + np.abs(b["hw"]) / HW_H0)
            L = float(np.log10(1.0 + 1e3 / HW_H0))
            print(f"water sheet colours: signed log, -1 km (red) .. 0 (white) .. 1 km (blue) (--hw-max H: linear 0 .. H);"
                  f" all frames: {np.nanmin(hv):.3g} .. {np.nanmax(hv):.3g} m")
            panels.append(("hwl", -L, L, "RdBu", "water sheet (m)", 0.645 if three else 1.0, "scene2"))
    if three and p3 == "melt":
        mv = np.concatenate([b["melt"] for _, _, b, _ in data])
        # the range of the melt where it is real: the grounded temperate bed (elsewhere it is 0 by definition or
        # round-off of the loads); white at 0, red freeze-on, blue melt
        sel = [b["melt"][(b["mask"] > 0.5) & (np.nan_to_num(b["ct"], nan=-1.0) > 0)]
               for _, _, b, _ in data if "mask" in b and "ct" in b]
        mt = np.concatenate(sel) if sel and sum(len(x) for x in sel) else mv
        if MELT_MAX:
            lo_, hi_ = -MELT_MAX, MELT_MAX
        else:
            # the 2nd .. 98th percentile there (a few extreme nodes would leave the rest of the strip near-white)
            lo_, hi_ = min(float(np.nanpercentile(mt, 2)), 0.0), max(float(np.nanpercentile(mt, 98)), 0.0)
            if hi_ - lo_ <= 0:
                lo_, hi_ = -1e-6, 1e-6
        z0 = -lo_ / (hi_ - lo_)
        cs_ = [[0.0, "#b2182b"], [z0, "#ffffff"], [1.0, "#08306b"]] if 0 < z0 < 1 else \
            ([[0.0, "#ffffff"], [1.0, "#08306b"]] if z0 <= 0 else [[0.0, "#b2182b"], [1.0, "#ffffff"]])
        q = np.nanpercentile(mt, [0, 10, 50, 90, 100]) if len(mt) else [np.nan] * 5
        print(f"melt on the grounded temperate bed ({len(mt)} node values): min {q[0]:.3g}, 10% {q[1]:.3g}, median "
              f"{q[2]:.3g}, 90% {q[3]:.3g}, max {q[4]:.3g} m/yr")
        print(f"melt rate colours {lo_:.3g} .. {hi_:.3g} m/yr: its 2nd .. 98th percentile there, all frames "
              f"(--melt-max M: -M .. M; red freeze-on, white 0, blue melt); everywhere {np.nanmin(mv):.3g} .. "
              f"{np.nanmax(mv):.3g} m/yr")
        panels.append(("melt", lo_, hi_, cs_, "melt rate (m/yr)", 0.99, "scene3"))
    if three and p3 == "ctm":
        panels.append(("ctm", -1.0, 1.0, [[0, "#3a7bd5"], [0.333, "#3a7bd5"], [0.334, "#bbbbbb"], [0.666, "#bbbbbb"],
                                          [0.667, "#c0392b"], [1, "#c0392b"]], "bed", 0.99, "scene3"))
    if COMPARE:                                   # the second run in the same scene, beside the first in y
        if CMP_FIELD == "T":                      # both coloured by temperature (one colour bar)
            panels = [("T", tmin, tmax, "RdBu_r", "temperature (K)", 1.0, "scene"),
                      ("cmp", tmin, tmax, "RdBu_r", "temperature (K)", 1.0, "scene")]
        else:
            panels.append(("cmp", 0.0, smax, SPEED_CS, "speed (m/yr)", 1.0, "scene"))

    W = max(float(np.ptp(tp["y"])) for _, tp, _, _ in data if tp is not None)      # strip width (m)
    CMP_SHIFT = 1.25 * W * TILE                   # --compare: the second run this far along y (a gap of W/4)

    def mesh_lines(base, P):
        """the element edges of the base (white), just below it, for every copy: the rows and columns of a structured
        footprint, the triangle edges of an unstructured one"""
        lx, ly, lz = [], [], []
        if structured(base["x"], base["y"]):
            xr, yr = np.round(base["x"], 0), np.round(base["y"], 0)
            for c in range(TILE):
                dy = c * W / 1e3
                for key, other in ((yr, xr), (xr, yr)):    # lines of constant y (along x), then of constant x
                    for v in np.unique(key):
                        m = np.where(key == v)[0]
                        o = m[np.argsort(other[m])]
                        lx += list(base["x"][o] / 1e3) + [None]
                        ly += list(base["y"][o] / 1e3 + dy) + [None]
                        lz += list(base["v"][o] - 3.0) + [None]
        else:
            if MESH_DIR:                          # the real elements (the Delaunay triangles are only for drawing)
                i, j, k = mesh_triangles(MESH_DIR, base["x"], base["y"])
            else:
                i, j, k = triangles(base["x"], base["y"])
                print("  base mesh: Delaunay triangles of the nodes, NOT the mesh's elements (--mesh-dir for those)")
            e = np.sort(np.r_[np.c_[i, j], np.c_[j, k], np.c_[k, i]], axis=1)
            e = np.unique(e, axis=0)
            n = len(e)
            for c in range(TILE):
                dy = c * W / 1e3
                X = np.full((n, 3), np.nan); Y = np.full((n, 3), np.nan); Z = np.full((n, 3), np.nan)
                X[:, :2] = base["x"][e] / 1e3; Y[:, :2] = base["y"][e] / 1e3 + dy; Z[:, :2] = base["v"][e] - 3.0
                if SECTOR:
                    Xr, Yr = tile_xy(list((base["x"][e] / 1e3).ravel()), list((base["y"][e] / 1e3).ravel()), c, W)
                    X[:, :2] = np.array(Xr, float).reshape(-1, 2); Y[:, :2] = np.array(Yr, float).reshape(-1, 2)
                lx += [None if np.isnan(v) else round(float(v), 3) for v in X.ravel()]
                ly += [None if np.isnan(v) else round(float(v), 3) for v in Y.ravel()]
                lz += [None if np.isnan(v) else round(float(v), 1) for v in Z.ravel()]
            print(f"  base mesh: {n} edges per copy")
        # white edges, dark grey on the speed panel (its slow ice is white)
        return go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="#444444" if P[0] in ("speed", "cmp") else "white", width=1),
                            name="base mesh", hoverinfo="skip", scene=P[6], showlegend=P[6] == "scene",
                            connectgaps=False)

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
        if SECTOR:                # the two arcs (the window's r ends) by angle, the two angled sides by r
            r, ph = np.hypot(xs, ys), np.arctan2(ys, xs)
            for re_ in (r.min(), r.max()):
                m = np.where(np.abs(r - re_) < 1.0)[0]
                wall(m[np.argsort(ph[m])], 0.0, "end wall")
            for m in sector_sides(xs, ys):            # ordered by radius (straight or jagged sides)
                wall(m, 0.0, "side wall")
            return out
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
        if SECTOR:                # copies rotated by the sector angle (velocity components turned with them)
            rot = lambda c, a, b: (np.cos(c * SECTOR_A) * a - np.sin(c * SECTOR_A) * b,
                                   np.sin(c * SECTOR_A) * a + np.cos(c * SECTOR_A) * b)
            out = {}
            for k in keys:
                parts = []
                for c in range(TILE):
                    if k in ("x", "y"):
                        X_, Y_ = rot(c, d["x"], d["y"])
                        parts.append(X_ if k == "x" else Y_)
                    elif k in ("vx", "vy"):
                        X_, Y_ = rot(c, d["vx"], d["vy"])
                        parts.append(X_ if k == "vx" else Y_)
                    else:
                        parts.append(d[k])
                out[k] = np.concatenate(parts)
        else:
            out = {k: np.concatenate([d[k] + (c * W if k == "y" else 0) for c in range(TILE)]) for k in keys}
        _, first = np.unique(np.c_[np.round(out["x"], 0), np.round(out["y"], 0)], axis=0, return_index=True)
        first = np.sort(first)
        return {k: v[first] for k, v in out.items()}

    def panel_traces(top, base, P):
        """TILE copies side by side, copy c shifted by c W in y: at each seam y = c W the image side (y = W) of
        one copy meets the master side (y = 0) of the next, so anything not periodic shows as a step there"""
        cmp = P[0] == "cmp"
        if cmp:                                   # the compared run: its own surfaces, drawn as speed, shifted in y
            if top is None or "_cmp" not in top:
                top, base = None, None
            else:                                 # shifted once and kept (the streamline cache goes by the dict)
                if "_cmp_sh" not in top:
                    sh = lambda d: None if d is None else {k: (v + CMP_SHIFT if k == "y" else v) for k, v in d.items()}
                    top["_cmp_sh"] = tuple(sh(d) for d in top["_cmp"])
                top, base = top["_cmp_sh"]
            P = (CMP_FIELD,) + tuple(P[1:])
        f, lo, hi, cs, ctitle, cx, sc = P
        unit = "K" if f == "T" else "m" if f in ("hw", "hwl") else "" if f == "ctm" else "m/yr"
        first_panel = sc == "scene" and not cmp
        tr = []
        # all copies as ONE surface each (the seam rows shared): separate surfaces are lit with one-sided normals at
        # their edges, which draws a faint line along a seam even where the values match exactly
        T_, B_ = tiled(top), tiled(base)
        # a bed field (water sheet, melt, mask): no surface, the base seen from above. Always: the surface VTU can
        # carry the same field (Melt Rate, 0 at the surface), and that surface then hid the bed and its hover read 0
        if T_ is not None and (f not in T_ or f in ("hw", "hwl", "ctm", "melt")):
            T_ = None
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
            if T_ is None and top is not None and f in ("hw", "hwl", "ctm", "melt"):   # a base-only panel: own colour bar
                cb = dict(title=ctitle, x=cx, len=0.8)
                if f == "hwl":                  # ticks at round thicknesses
                    tv = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
                    cb.update(tickvals=[float(np.sign(v) * np.log10(1 + abs(v) / HW_H0)) for v in [-x for x in tv[:0:-1]] + tv],
                              ticktext=[f"{v:g}" for v in [-x for x in tv[:0:-1]] + tv])
                if f == "ctm":
                    cb.update(tickvals=[-1, 0, 1], ticktext=["cold", "floating", "temperate"])
                # no shading: lit, the colours came out greyed by the angle to the light (a flat colour map here)
                col = dict(intensity=B_[f], colorscale=cs, cmin=lo, cmax=hi, showscale=True, colorbar=cb,
                           lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0))
            elif f != "speed" or BASE_COLOUR == "speed":
                col = dict(intensity=B_[f], colorscale=cs, cmin=lo, cmax=hi, showscale=False)
            else:
                col = dict(intensity=np.where(mask > -0.5, 1.0, 0.0), cmin=0, cmax=1,
                           colorscale=[[0, "#3a7bd5"], [1, "#c0392b"]], showscale=False)
            # a base-only panel (sheet, mask) is a map: drawn flat just below the lowest base, so no part of the base
            # (the shelf base above the grounded bed) hides another as the view turns; the lines stay above it
            zb_ = np.full(len(B_["v"]), ZFLAT) if (T_ is None and f in ("hw", "hwl", "ctm", "melt")) else B_["v"]
            tr.append(go.Mesh3d(x=B_["x"] / 1e3, y=B_["y"] / 1e3, z=zb_, i=i, j=j, k=k, **col,
                                customdata=np.stack([B_["hw" if f == "hwl" else f], mask], 1), scene=sc,
                                name="base Zb", opacity=1.0, showlegend=False,
                                hovertemplate="x %{x:.2f} km, y %{y:.2f} km<br>Zb %{z:.1f} m<br>"
                                              f"{'bed T' if f == 'T' else 'cold -1 / floating 0 / temperate 1:' if f == 'ctm' else 'hw' if f == 'hwl' else f} %{{customdata[0]:{'.3g' if f in ('hw', 'hwl', 'melt') else '.0f' if f == 'ctm' else '.2f'}}} {unit}"
                                              "<br>mask %{customdata[1]}<extra></extra>"))
            if SHOW_BED and "bed" in B_:
                tr.append(go.Mesh3d(x=B_["x"] / 1e3, y=B_["y"] / 1e3, z=B_["bed"] - 1.0, i=i, j=j,
                                    k=k, color="#999999", opacity=0.35, name="bedrock", hoverinfo="skip",
                                    showlegend=False, scene=sc))
        if SHOW_PARTS and base is not None and top is not None:
            t_, p_ = mesh_elements(MESH_DIR, base["x"], base["y"])
            e = partition_edges(t_, p_)
            tzd = {(round(x, 0), round(y, 0)): v for x, y, v in zip(top["x"], top["y"], top["v"])}
            zt_ = np.array([tzd.get((round(x, 0), round(y, 0)), v) for x, y, v in
                            zip(base["x"], base["y"], base["v"])])
            for zz, off, show in ((zt_, 8.0, first_panel), (base["v"], -6.0, False)):
                lx, ly, lz = [], [], []
                for c in range(TILE):
                    n = len(e)
                    X = np.full((n, 3), np.nan); Y = np.full((n, 3), np.nan); Z = np.full((n, 3), np.nan)
                    X[:, :2] = base["x"][e] / 1e3; Y[:, :2] = base["y"][e] / 1e3 + c * W / 1e3
                    if SECTOR:
                        Xr, Yr = tile_xy(list((base["x"][e] / 1e3).ravel()), list((base["y"][e] / 1e3).ravel()), c, W)
                        X[:, :2] = np.array(Xr, float).reshape(-1, 2); Y[:, :2] = np.array(Yr, float).reshape(-1, 2)
                    Z[:, :2] = zz[e] + off
                    lx += [None if np.isnan(v) else round(float(v), 3) for v in X.ravel()]
                    ly += [None if np.isnan(v) else round(float(v), 3) for v in Y.ravel()]
                    lz += [None if np.isnan(v) else round(float(v), 1) for v in Z.ravel()]
                tr.append(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="#444444" if f == "speed" else "white", width=3),
                                       name=f"partitions ({len(np.unique(p_))} in the window)",
                                       legendgroup="parts", hoverinfo="skip", scene=sc, showlegend=show,
                                       connectgaps=False))
        if SIDES and top is not None and base is not None and T_ is not None:
            tr += walls(T_, B_, P) if SECTOR else walls(top, base, P)
        if base is not None and "mask" in base:      # both panels (empty without a mask: same traces per state)
            # the cold/temperate boundary of the grounded bed (ColdTempMask +1 temperate, -1 cold; floating counted
            # cold), magenta, on the surface and the base of both panels; hidden until "temperate on" (MASK_IDX)
            tzd = {(round(x, 0), round(y, 0)): v for x, y, v in zip(top["x"], top["y"], top["v"])} if top else {}
            zt_ = np.array([tzd.get((round(x, 0), round(y, 0)), v) for x, y, v in zip(base["x"], base["y"], base["v"])])
            if "ct" in base:
                fm = np.where(base["mask"] > -0.5, np.nan_to_num(base["ct"], nan=-1.0), -1.0)
                mx, my, mb, mt = contour(base["x"], base["y"], fm, 0.0, [base["v"], zt_])
            else:
                mx, my, mb, mt = [], [], [], []
            qx, qy, qt, qb = [], [], [], []
            for c in range(TILE):
                X_, Y_ = tile_xy(mx, my, c, W)
                qx += X_ + [None]
                qy += Y_ + [None]
                qt += [None if v is None else v + 6.0 for v in mt] + [None]
                qb += [None if v is None else v - 6.0 for v in mb] + [None]
            for z_, nm in ((qt, "cold | temperate bed (on the surface)"), (qb, "cold | temperate bed (on the base)")):
                tr.append(go.Scatter3d(x=qx, y=qy, z=z_, mode="lines", line=dict(color="#ff00e0", width=6), name=nm,
                                       legendgroup="ctmask", scene=sc, showlegend=False, connectgaps=False,
                                       hovertemplate="cold | temperate at x %{x:.2f} km, y %{y:.2f} km<extra></extra>"))
        if base is not None and "mask" in base:
            # the grounding line through all copies (a jump at a seam: not periodic), in red on the surface and on
            # the base (just outside each, so neither surface hides it)
            gx, gy, gzb, gzt = gl_line(base, top)
            ax, ay, at, ab = [], [], [], []
            add = lambda v, a: None if v is None else v + a
            for c in range(TILE):
                X_, Y_ = tile_xy(list(gx), list(gy), c, W)
                ax += X_ + [None]
                ay += Y_ + [None]
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
                    X_, Y_ = tile_xy(cx, cy, c, W)
                    bx += X_ + [None]
                    by += Y_ + [None]
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
                        once(f"  full-Stokes | SSA boundary: x {min(px):.2f} .. {max(px):.2f} km")
        if STREAM and not DIFF and top is not None and "vx" in top and "vy" in top:
            if SECTOR:            # traced unrolled (r, arc), the copies rotated
                sx, sy, sz = streamlines_sector(top, STREAM_KM)
                cp = [np.c_[np.cos(c * SECTOR_A) * sx - np.sin(c * SECTOR_A) * sy,
                            np.sin(c * SECTOR_A) * sx + np.cos(c * SECTOR_A) * sy].astype(np.float32) for c in range(TILE)]
                qx = np.concatenate([q[:, 0] for q in cp])
                qy = np.concatenate([q[:, 1] for q in cp])
            else:
                sx, sy, sz = streamlines(top, STREAM_KM, W, periodic=not WALLS)
                qx = np.concatenate([sx] * TILE)
                qy = np.concatenate([sy + np.float32(c * W / 1e3) for c in range(TILE)])
            qz = np.concatenate([sz] * TILE)
            tr.append(go.Scatter3d(x=qx, y=qy, z=qz, mode="lines", connectgaps=False,
                                   line=dict(color="white", width=2),
                                   name="streamlines (surface velocity)", legendgroup="stream", hoverinfo="skip",
                                   scene=sc, showlegend=first_panel))
        if top is not None and TILE > 1:
            if SECTOR:                # side 2 (the largest angle) of copy c - 1 = side 1 of copy c
                m = sector_sides(top["x"], top["y"])[1]
                o = np.arange(len(m))
            else:
                m = np.abs(top["y"] - W) < 1
                o = np.argsort(top["x"][m])
            sx, sz = top["x"][m][o] / 1e3, top["v"][m][o] + 15.0
            sy0 = top["y"][m][o] / 1e3
            qx, qy, qz = [], [], []
            for c in range(1, TILE):
                if SECTOR:
                    X_, Y_ = tile_xy(list(sx), list(sy0), c - 1, W)
                else:
                    X_, Y_ = list(sx), [c * W / 1e3] * len(sx)
                qx += X_ + [None]
                qy += Y_ + [None]
                qz += list(sz) + [None]
            tr.append(go.Scatter3d(x=qx, y=qy, z=qz, mode="lines",
                                   line=dict(color="black", width=3, dash="dash"),
                                   name="periodic seam (y = W of one copy = y = 0 of the next)", hoverinfo="skip",
                                   scene=sc, showlegend=first_panel))
        return tr
    # every state's traces, ALIGNED: plotly matches a frame's traces to the figure's by position, so a state with a
    # trace another lacks (cold/temperate boundary lines once part of the bed is temperate, a grounding-line piece,
    # the FS | SSA line) would shift every later trace by one: e.g. its streamlines landing in a slot the toggle does
    # not cover (a second, uncontrollable layer). Each trace keyed by (type, name, legend group, scene, n-th of that
    # key); the union of the keys in order of first appearance; a state without one gets an empty trace of that type.
    per = [traces(tp, b) for _, tp, b, _ in data]
    def tkey(t_):
        return (t_.type, getattr(t_, "name", None), getattr(t_, "legendgroup", None), getattr(t_, "scene", None))
    slots, keyed = [], []
    for tr in per:
        cnt, ks = {}, []
        for t_ in tr:
            k = tkey(t_)
            cnt[k] = cnt.get(k, 0) + 1
            ks.append(k + (cnt[k],))
        keyed.append(dict(zip(ks, tr)))
        for k in ks:
            if k not in slots:
                slots.append(k)

    def blank(k):
        typ, nm, lg, sc_ = k[:4]
        kw = dict(name=nm, scene=sc_, showlegend=False, hoverinfo="skip")
        if lg is not None:
            kw["legendgroup"] = lg
        if typ == "mesh3d":
            return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], **kw)
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", **kw)
    per = [[kd[k] if k in kd else blank(k) for k in slots] for kd in keyed]
    if any(len(kd) != len(slots) for kd in keyed):
        print(f"  traces aligned over the states: {len(slots)} slots "
              f"({sum(len(slots) - len(kd) for kd in keyed)} empty fillers)")
    frame0 = [t_.__class__(t_) for t_ in per[0]]      # the first state's frame: copies (the figure's own traces get
    first = per[0]                                    # "visible" settings below; frames never set it)
    # the cold/temperate boundary toggle ("temperate off" / "temperate on"): those traces are in every frame (they move
    # with the slider) but hidden in the figure until switched on; the frames do not set "visible", so the choice stays
    ct_idx = [n for n, t_ in enumerate(first) if getattr(t_, "legendgroup", None) == "ctmask"]
    for n in ct_idx:
        first[n].visible = False
    # the colour switch of the velocity panel: speed, vx, vy, vz. The slider's frames draw the panel's own field; the
    # buttons and the slider share one script (switch_js) that keeps the chosen field and puts that field of the state
    # on show into the surfaces and walls, so a button acts on the current state and the choice survives the slider.
    # Colour ranges over all states.
    tp0, b0 = data[0][1], data[0][2]
    menus, switch_js = [], None
    if SWITCH and tp0 is not None and b0 is not None and "vz" in tp0:
        from plotly.colors import get_colorscale
        P0 = panels[0]
        ref = panel_traces(tp0, b0, P0)
        get = lambda t_, k: t_[k] if k in t_ else None
        idx = [n for n, t_ in enumerate(ref) if get(t_, "intensity") is not None]
        allv = lambda fld: np.concatenate([d[fld] for _, tp, b, _ in data for d in (tp, b) if d is not None and fld in d])
        ranges, vals = {}, [dict() for _ in data]
        for fld, title in (("speed", "speed (m/yr)"), ("vx", "vx, along the flow (m/yr)"),
                           ("vy", "vy, across (m/yr)"), ("vz", "vz, vertical (m/yr)")):
            if fld == "speed" and not DIFF:
                lo, hi, cs = P0[1], P0[2], P0[3]
            else:
                v = np.abs(allv(fld))
                v = v[np.isfinite(v)]
                hi = float(np.percentile(v, 99)) if len(v) else 1.0
                if fld == "vx" and not DIFF:
                    hi = P0[2]
                hi = max(hi, 1e-6)
                lo, cs = -hi, "RdBu_r"
            ranges[fld] = dict(lo=lo, hi=hi, cs=cs if isinstance(cs, list) else get_colorscale(cs), title=title)
            for n, (_, tp, b, _) in enumerate(data):
                tf = [t_ for t_ in panel_traces(tp, b, (fld, lo, hi, cs, title, P0[5], P0[6]))
                      if get(t_, "intensity") is not None]
                if len(tf) != len(idx):              # a state drawn with other traces: no switch there
                    vals[n][fld] = None
                    continue
                vals[n][fld] = [base64.b64encode(np.asarray(t_["intensity"], dtype="<f4").tobytes()).decode()
                                for t_ in tf]           # float32, base64 (decoded in the page: dec())
            print(f"  colour switch {fld}: {lo:.3g} .. {hi:.3g} m/yr")
        buttons = [dict(label=fld, method="skip", args=[None]) for fld in ranges]
        menus = [dict(type="buttons", direction="right", buttons=buttons, x=0.0, y=1.06, xanchor="left",
                      yanchor="bottom", showactive=True, pad=dict(r=4, t=0),
                      active=list(ranges).index(P0[0]) if P0[0] in ranges else 0)]
        switch_js = ("""
var gdv = document.getElementsByClassName('plotly-graph-div')[0];
var V = %s, R = %s, IDX = %s, OWN = %s, cur = {s: 0, f: OWN};
function dec(b) { var s = atob(b), u = new Uint8Array(s.length);
  for (var i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return Array.from(new Float32Array(u.buffer)); }
function showField() {
  var e = V[cur.s][cur.f], r = R[cur.f], cd = [], d = [];
  if (!e) return;
  for (var q = 0; q < IDX.length; q++) {
    var old = gdv.data[IDX[q]].customdata, a = dec(e[q]);
    d.push(a);
    if (old && old.length && Array.isArray(old[0])) cd.push(a.map(function (x, k) { return [x, old[k][1]]; }));
    else cd.push(a);
  }
  var n = IDX.length, u = {intensity: d, customdata: cd, cmin: Array(n).fill(r.lo), cmax: Array(n).fill(r.hi),
           colorscale: Array(n).fill(r.cs)};
  u['colorbar.title.text'] = Array(n).fill(r.title);
  Plotly.restyle(gdv, u, IDX);
}
gdv.on('plotly_buttonclicked', function (e) { if (e.button && R[e.button.label]) { cur.f = e.button.label; showField(); } });
gdv.on('plotly_sliderchange', function (e) { cur.s = parseInt(e.step.args[0][0]); });
gdv.on('plotly_animated', function () { if (cur.f !== OWN) showField(); });
""" % (json.dumps(vals, allow_nan=False), json.dumps(ranges), json.dumps(idx), json.dumps(P0[0])))
    def tidy(tr):
        """compact copies of the big arrays: plotly >= 6 writes numpy arrays as binary (base64 of their dtype), so
        float32 / int32 halve them; plotly < 6 writes every float as text in full (~18 characters), so rounded:
        positions to 1 m, heights to 0.1 m, colours to 0.01"""
        import plotly
        f32 = int(plotly.__version__.split(".")[0]) >= 6
        cast = (lambda a: a.astype(np.float32)) if f32 else (lambda a: a)
        for t_ in tr:
            if getattr(t_, "type", None) != "mesh3d" or t_.x is None or len(t_.x) < 1000:
                continue
            t_.x = cast(np.round(np.asarray(t_.x, float), 3)); t_.y = cast(np.round(np.asarray(t_.y, float), 3))
            t_.z = cast(np.round(np.asarray(t_.z, float), 1))
            for k in ("i", "j", "k"):
                if t_[k] is not None:
                    t_[k] = np.asarray(t_[k], dtype=np.int32)
            for k in ("intensity", "customdata"):
                if t_[k] is not None:
                    t_[k] = cast(np.round(np.asarray(t_[k], float), 2))
        return tr
    tidy(first)

    def lean(tr):
        """a frame's traces: the surfaces (Mesh3d with the first state's x, y and triangles: the same mesh) as their
        z and colours only, so the page holds the geometry once, not once per state"""
        out = []
        for q, t_ in enumerate(tr):
            f0 = first[q] if q < len(first) else None
            same = (f0 is not None and t_.type == "mesh3d" and f0.type == "mesh3d" and t_.i is not None
                    and len(t_.x) == len(f0.x) and len(t_.i) == len(f0.i)
                    and all(np.array_equal(np.asarray(t_[k]), np.asarray(f0[k])) for k in ("x", "y", "i", "j", "k")))
            if same:
                d = dict(type="mesh3d", z=t_.z)
                for k in ("intensity", "customdata", "cmin", "cmax"):
                    if t_[k] is not None:
                        d[k] = t_[k]
                out.append(d)
            else:
                out.append(t_)
        return out
    # the base mesh (white element edges) as a toggle: one trace per panel AFTER the frames' traces (the slider never
    # touches them: drawn at the first state's base), hidden unless --mesh; buttons "mesh off" / "mesh on"
    if ct_idx:
        menus = menus + [dict(type="buttons", direction="right", x=0.72, y=1.06, xanchor="left", yanchor="bottom",
                              showactive=True, pad=dict(r=4, t=0), active=0,
                              buttons=[dict(label="temperate off", method="restyle", args=[{"visible": False}, ct_idx]),
                                       dict(label="temperate on", method="restyle", args=[{"visible": True}, ct_idx])])]
        print(f"  cold/temperate boundary toggle on {len(ct_idx)} traces (magenta, both panels)")
    st_idx = [n for n, t_ in enumerate(first) if getattr(t_, "legendgroup", None) == "stream"]
    if st_idx:
        menus = menus + [dict(type="buttons", direction="right", x=0.84, y=1.06, xanchor="left", yanchor="bottom",
                              showactive=True, pad=dict(r=4, t=0), active=1,
                              buttons=[dict(label="streamlines off", method="restyle", args=[{"visible": False}, st_idx]),
                                       dict(label="streamlines on", method="restyle", args=[{"visible": True}, st_idx])])]
        print(f"  streamlines of the surface velocity: seeds every {STREAM_KM:g} km across (--stream-spacing, "
              f"--no-streamlines)")
    if MESH_TOGGLE and data[0][2] is not None:
        midx = []
        for P in panels:
            t_ = mesh_lines(data[0][2], P)
            t_.visible = SHOW_MESH
            midx.append(len(first))
            first.append(t_)
        menus = menus + [dict(type="buttons", direction="right", x=0.52 if two else 0.40, y=1.06, xanchor="left",
                              yanchor="bottom", showactive=True, pad=dict(r=4, t=0), active=1 if SHOW_MESH else 0,
                              buttons=[dict(label="mesh off", method="restyle", args=[{"visible": False}, midx]),
                                       dict(label="mesh on", method="restyle", args=[{"visible": True}, midx])])]
        print(f"  base mesh toggle on {len(midx)} panel(s) (drawn at the first state's base)")
    frames = [go.Frame(data=lean(tidy(per[n] if n else frame0)), name=str(n))
              for n in range(len(data))]
    steps = [dict(method="animate", label=l, args=[[str(n)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                                  transition=dict(duration=0))])
             for n, (l, _, _, _) in enumerate(data)]
    zs = [v for _, tp, b, _ in data for d in (tp, b) if d is not None for v in (np.nanmin(d["v"]), np.nanmax(d["v"]))]
    zs += [np.nanmin(b["bed"]) for _, _, b, _ in data if b is not None and "bed" in b]
    asp = dict(ASPECT)
    asp["z"] = ASPECT["z"] * Z_EXAGGERATION
    asp["y"] = ASPECT["y"] * TILE * (2.25 if COMPARE else 1.0)
    if SECTOR:                    # the copies' true shape: x and y on ONE scale (a different stretch in y than in x
        # shears the rotated copies), over the extent of all copies
        allx = np.concatenate([tp["x"] for _, tp, _, _ in data if tp is not None])
        ally = np.concatenate([tp["y"] for _, tp, _, _ in data if tp is not None])
        tx_, ty_ = [], []
        for c in range(TILE):
            ca, sa = np.cos(c * SECTOR_A), np.sin(c * SECTOR_A)
            tx_.append(ca * allx - sa * ally); ty_.append(sa * allx + ca * ally)
        tx_, ty_ = np.concatenate(tx_), np.concatenate(ty_)
        SECTOR_XR = [float(tx_.min()) / 1e3 - 1.0, float(tx_.max()) / 1e3 + 1.0]
        SECTOR_YR = [float(ty_.min()) / 1e3 - 1.0, float(ty_.max()) / 1e3 + 1.0]
        asp["y"] = asp["x"] * (SECTOR_YR[1] - SECTOR_YR[0]) / (SECTOR_XR[1] - SECTOR_XR[0])
    fig = go.Figure(data=first, frames=frames)
    xr_ = [x0, x1]
    yax = dict(title="y (km)")
    if SECTOR:                    # the window is in r: the axes span all the copies
        xr_ = SECTOR_XR
        yax["range"] = SECTOR_YR
    scene = dict(xaxis=dict(title="x (km)", range=xr_), yaxis=yax,
                 zaxis=dict(title="z (m)", range=[min(zs) - 20, max(zs) + 20]),
                 aspectmode="manual", aspectratio=asp)
    scenes = {"scene": dict(scene, domain=dict(x=[0.0, 0.29] if three else [0.0, 0.46] if two else [0.0, 1.0],
                                               y=[0.0, 1.0]))}
    if two:
        scenes["scene2"] = dict(scene, domain=dict(x=[0.345, 0.635] if three else [0.52, 0.98], y=[0.0, 1.0]))
    if three:
        scenes["scene3"] = dict(scene, domain=dict(x=[0.69, 0.98], y=[0.0, 1.0]))
    ann = ([dict(text=t_, x=x_, y=1.0, xref="paper", yref="paper", showarrow=False, font=dict(size=16))
            for t_, x_ in (zip(("temperature" if sheetT else "velocity", "water sheet", "melt rate" if p3 == "melt" else "cold / temperate bed"), (0.145, 0.49, 0.835)) if three else
                           zip(("velocity", "water sheet" if sheet else "temperature"), (0.23, 0.75)))]
           if two else [])
    if COMPARE:                                   # the two runs named above their ice sheets, in the scene
        ztop = max(zs) + 20
        scenes["scene"]["annotations"] = [
            dict(x=x0, y=W * TILE / 2e3, z=ztop, text=os.path.basename(str(items[0][1]))[:-4], showarrow=False,
                 font=dict(size=14), xanchor="left"),
            dict(x=x0, y=(CMP_SHIFT + W * TILE / 2) / 1e3, z=ztop, text=os.path.basename(COMPARE[0])[:-4],
                 showarrow=False, font=dict(size=14), xanchor="left")]
    # play / pause: through the states in turn from the one on show (PLAY_MS per state, --play-ms)
    anim = dict(frame=dict(duration=PLAY_MS, redraw=True), transition=dict(duration=0), mode="immediate",
                fromcurrent=True)
    play = dict(type="buttons", direction="left", x=0.1, y=0, xanchor="right", yanchor="top", showactive=False,
                pad=dict(r=10, t=55),
                buttons=[dict(label="\u25b6 play", method="animate", args=[None, anim]),
                         dict(label="\u23f8 pause", method="animate",
                              args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                 transition=dict(duration=0))])])
    fig.update_layout(
        title=f"{sys.argv[1] if len(sys.argv) > 1 else ''}: {data[0][0]}", **scenes, annotations=ann,
        sliders=[dict(steps=steps, currentvalue=dict(prefix="state: "), pad=dict(t=30), x=0.1, len=0.9)],
        autosize=True, template="plotly_white", legend=dict(x=0, y=1), updatemenus=menus + [play],
        margin=dict(l=10, r=10, t=60, b=10))
    out = os.path.splitext(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "plot_state_3d")[0] + "_3d.html"
    sync = ("""
var gd = document.getElementsByClassName('plotly-graph-div')[0], busy = false;
gd.on('plotly_relayout', function (e) {
  if (busy) return;
  var u = {}, sc = ['scene', 'scene2', 'scene3'];
  sc.forEach(function (a) { if (e[a + '.camera']) sc.forEach(function (b) {
    if (b != a && gd.layout[b]) u[b + '.camera'] = e[a + '.camera']; }); });
  if (Object.keys(u).length) { busy = true; Plotly.relayout(gd, u).then(function () { busy = false; }); }
});""" if two else None)
    post = "\n".join(x for x in (sync, switch_js) if x) or None
    # the page width and height (no fixed size: no sideways scrolling; resizes with the window)
    fig.write_html(out, include_plotlyjs="cdn", post_script=post, config=dict(responsive=True),
                   default_width="100%", default_height="95vh")
    print(f"wrote {out}")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
