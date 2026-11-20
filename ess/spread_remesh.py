"""
Remeshing the spreading test ([case] geometry = "spread", branch test/lava-spreading) to follow the current's margin.

The margin is taken from the state directly: the ice thickness H = Zs - bed at every footprint node, its contour at
H = spread_thickness + margin_dh (the current: thicker than the background layer by margin_dh). The new footprint (the
same rectangle, walls on all four sides, physical groups as ess/case.py slab_geo) gets the element size

    h = fine                                   within band of the margin,
        fine + grade (d - band)                farther out (d: the distance to the margin), up to
        coarse_in                              inside the current (H above the margin's value: the dome; coarse here
                                               also relaxes the free surface's time-step limit, which scales as
                                               (element size)^2 x thickness^3)
        coarse_out                             on the background layer outside (default 4 x spread_res: nothing
                                               happens there until the margin arrives, and the band's grading
                                               refines ahead of it),

graded (|grad h| <= grade) and written as a gmsh background view (as ess/remesh.py). Then ElmerGrid, partitioned for
n_procs (-metiskway), and the state carried over level by level (ess/remesh.py interpolate_state: linear in the old
triangles; the flat bed exact).

needs_remesh: the margin's points against the CURRENT mesh's local size (the mean edge length at the nearest node):
remesh when any margin point sits where the mesh is coarser than trigger x fine (the margin is leaving its band).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

import numpy as np

from .remesh import _edges, grade, interpolate_state, write_pos
from .remesh import read_footprint as _read_serial
from .statefile import SavedState, find_name

log = logging.getLogger("ess")


def read_footprint(meshdir):
    """(ids, x, y, tri) of an Elmer 2D footprint: its serial files (mesh.nodes, mesh.elements), or, when ElmerGrid kept
    only the partitioned ones (partitioning.N/part.*.nodes, part.*.elements: global node ids), their union"""
    if os.path.exists(os.path.join(meshdir, "mesh.nodes")):
        return _read_serial(meshdir)
    import glob
    parts = sorted(glob.glob(os.path.join(meshdir, "partitioning.*")))
    if not parts:
        raise FileNotFoundError(f"{meshdir}: neither mesh.nodes nor a partitioning.* directory")
    pd = parts[0]
    nid, nx, ny = [], [], []
    empty = []
    for f in glob.glob(os.path.join(pd, "part.*.nodes")):
        if os.path.getsize(f) == 0:
            empty.append(os.path.basename(f))
            continue
        a = np.loadtxt(f, ndmin=2)
        if a.shape[1] < 4:
            raise SystemExit(f"{f}: {a.shape[1]} columns (expected id, partition, x, y, z)")
        nid.append(a[:, 0].astype(np.int64)); nx.append(a[:, 2]); ny.append(a[:, 3])
    if empty:
        log.warning(f"spread remesh: {len(empty)} of the partitions of {os.path.basename(meshdir)} have no nodes "
                    f"(e.g. {empty[0]}): too few footprint nodes for the number of ranks")
    nid, nx, ny = np.concatenate(nid), np.concatenate(nx), np.concatenate(ny)
    ids, k = np.unique(nid, return_index=True)
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise SystemExit(f"{pd}: node ids are not 1..n")
    x, y = nx[k], ny[k]
    el = {}
    for f in glob.glob(os.path.join(pd, "part.*.elements")):
        for line in open(f):
            w = line.split()
            if len(w) >= 6 and w[2] == "303":
                el[int(w[0])] = (int(w[3]), int(w[4]), int(w[5]))
    if not el:
        raise SystemExit(f"{pd}: no triangles (303) in part.*.elements")
    tri = np.array([el[k_] for k_ in sorted(el)], np.int64) - 1
    log.info(f"spread remesh: footprint {os.path.basename(meshdir)} read from {os.path.basename(pd)} "
             f"({len(ids)} nodes, {len(tri)} triangles)")
    return ids, x, y, tri


def default_opt(c):
    """the remeshing settings from [case] (spread_remesh_* keys, defaults below)"""
    res = float(c.get("spread_res", 5000.0))
    Ls, Ws = float(c["spread_length_km"]) * 1e3, float(c["width"])
    src = (dict(src_x=float(c.get("source_x_km", Ls / 2e3)) * 1e3, src_y=float(c.get("source_y_km", Ws / 2e3)) * 1e3,
                src_r=float(c.get("source_radius_km", 100.0)) * 1e3)
           if float(c.get("source_rate", 0.1)) > 0 else {})      # no source (a drop test): nothing at the centre
    return dict(**src,
                fine=float(c.get("spread_remesh_fine", 2000.0)),
                band=float(c.get("spread_remesh_band_km", 10.0)) * 1e3,
                # the vent: its own size (default the margin's) and band; the dome is thickest over it, where the
                # free surface's time-step limit ~ (element size)^2 / thickness^3 bites first
                vent_fine=float(c.get("spread_remesh_vent_fine", c.get("spread_remesh_fine", 2000.0))),
                vent_band=float(c.get("spread_remesh_vent_band_km", c.get("spread_remesh_band_km", 10.0))) * 1e3,
                grade=float(c.get("spread_remesh_grade", 0.2)),
                coarse_in=float(c.get("spread_remesh_coarse_in", 2 * res)),
                coarse_out=float(c.get("spread_remesh_coarse_out", 4 * res)),   # the untouched background layer: coarse
                trigger=float(c.get("spread_remesh_trigger", 1.5)),
                min_nodes_per_rank=int(c.get("spread_remesh_min_nodes_per_rank", 40)),
                Hc=float(c.get("spread_thickness", 100.0)) + float(c.get("margin_dh", 10.0)))


def thickness(src: SavedState, nb):
    """H = Zs - bed at the nb footprint nodes (node ids: level * nb + footprint id)"""
    blk = {b.name.lower(): b for b in src.layout.blocks}
    g = blk["zs"].gids
    zs = src.layout.split(src.x)[blk["zs"].name]
    H = np.full(nb, np.nan)
    gb, vb = src.frozen[find_name(src.frozen, "bedrock")]
    bed = np.full(nb, np.nan)
    lv = (gb - 1) // nb
    m = lv == lv.min()
    bed[(gb[m] - 1) % nb] = vb[m]
    H[(g - 1) % nb] = zs - bed[(g - 1) % nb]
    return H


def margin_points(x, y, tri, H, Hc, step=None):
    """points along the contour H = Hc on the footprint (densified to about step m), (n x 2); empty if none"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    if not (np.nanmax(H) > Hc > np.nanmin(H)):
        return np.zeros((0, 2))
    cs = plt.tricontour(mtri.Triangulation(x, y, tri), H, levels=[Hc])
    segs = [s for s in cs.allsegs[0] if len(s) > 1]
    plt.close("all")
    if not segs:
        return np.zeros((0, 2))
    out = []
    for s in segs:
        for a, b in zip(s[:-1], s[1:]):
            n = max(1, int(np.ceil(np.hypot(*(b - a)) / (step or 200.0))))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            out.append(a + t * (b - a))
        out.append(s[-1:])
    return np.vstack(out)


def local_size(x, y, tri):
    """the mean length of each node's edges"""
    e = _edges(tri)
    L = np.hypot(x[e[:, 0]] - x[e[:, 1]], y[e[:, 0]] - y[e[:, 1]])
    s, c = np.zeros(len(x)), np.zeros(len(x))
    for k in (0, 1):
        np.add.at(s, e[:, k], L)
        np.add.at(c, e[:, k], 1)
    return s / np.maximum(c, 1)


def needs_remesh(cfg, src: SavedState, mesh: str, opt=None):
    """(True/False, note): the margin where the current mesh is coarser than trigger x fine"""
    c = cfg["case"]
    opt = opt or default_opt(c)
    ids, x, y, tri = read_footprint(os.path.join(cfg["paths"]["case_dir"], mesh))
    H = thickness(src, len(ids))
    P = margin_points(x, y, tri, H, opt["Hc"], step=opt["fine"] / 2)
    if not len(P):
        return False, "no margin (H nowhere above the margin's value)"
    from scipy.spatial import cKDTree
    h = local_size(x, y, tri)
    _, k = cKDTree(np.c_[x, y]).query(P)
    hm = h[k]
    bad = hm > opt["trigger"] * opt["fine"]
    r = np.hypot(P[:, 0] - x.mean(), P[:, 1] - y.mean())
    note = (f"margin {len(P)} points, r {r.min() / 1e3:.1f} .. {r.max() / 1e3:.1f} km from the box centre; local mesh "
            f"size there {np.median(hm):.0f} m median, {hm.max():.0f} m max (remesh above "
            f"{opt['trigger'] * opt['fine']:.0f} m: {int(bad.sum())} points)")
    return bool(bad.any()), note


def size_field(x, y, tri, H, opt):
    """(h at the old footprint nodes, margin points, report)"""
    from scipy.spatial import cKDTree
    P = margin_points(x, y, tri, H, opt["Hc"], step=opt["fine"] / 2)
    nm = len(P)
    inside = H > opt["Hc"]
    if "src_r" in opt:
        inside = inside | (np.hypot(x - opt["src_x"], y - opt["src_y"]) < opt["src_r"])
    far = np.where(inside, opt["coarse_in"], opt["coarse_out"])
    h = far.astype(float)
    if len(P):
        d = cKDTree(P).query(np.c_[x, y])[0]
        h = np.minimum(opt["fine"] + opt["grade"] * np.maximum(d - opt["band"], 0.0), h)
    if "src_r" in opt:
        # the source (vent) always resolved, at its own size: vent_fine within vent_band of its rim and over its disc
        # (before the current has a margin of its own, and however close the margin is)
        dv = np.maximum(np.hypot(x - opt["src_x"], y - opt["src_y"]) - opt["src_r"], 0.0)
        vf = opt.get("vent_fine", opt["fine"])
        h = np.minimum(vf + opt["grade"] * np.maximum(dv - opt.get("vent_band", opt["band"]), 0.0), h)
    h = grade(h, x, y, tri, opt["grade"])
    n_est = np.sum(_areas(x, y, tri) / (0.433 * h ** 2))
    rep = (f"size: {opt['fine']:g} m within {opt['band'] / 1e3:g} km of the margin (H = {opt['Hc']:g} m; "
           f"{nm} points)" + (f"; the vent (r = {opt['src_r'] / 1e3:g} km) {opt.get('vent_fine', opt['fine']):g} m within "
           f"{opt.get('vent_band', opt['band']) / 1e3:g} km of its rim" if "src_r" in opt else "") + f"; grading {opt['grade']:g}, {opt['coarse_in']:g} m inside the current, "
           f"{opt['coarse_out']:g} m outside; ~{n_est:.0f} triangles")
    return h, P, rep


def _areas(x, y, tri):
    a = np.zeros(len(x))
    A = 0.5 * np.abs((x[tri[:, 1]] - x[tri[:, 0]]) * (y[tri[:, 2]] - y[tri[:, 0]])
                     - (x[tri[:, 2]] - x[tri[:, 0]]) * (y[tri[:, 1]] - y[tri[:, 0]]))
    for k in range(3):
        np.add.at(a, tri[:, k], A / 3)
    return a


def box_geo(L, W, pos_name):
    """gmsh .geo of the box 0 <= x <= L, 0 <= y <= W, the size from the background view; physical groups as
    ess/case.py slab_geo (inflow x = 0 22, front x = L 23, side1 y = 0 24, side2 y = W 25)"""
    return "\n".join([
        "// generated by ess/spread_remesh.py: the spreading test's box, the size from the state's margin",
        "Mesh.Algorithm = 5;", "Mesh.MeshSizeFromPoints = 0;", "Mesh.MeshSizeExtendFromBoundary = 0;",
        "Mesh.MeshSizeFromCurvature = 0;", f'Merge "{pos_name}";',
        "Field[1] = PostView;", "Field[1].ViewIndex = 0;", "Background Field = 1;",
        "Point(1) = {0, 0, 0, 1000};", f"Point(2) = {{{L!r}, 0, 0, 1000}};",
        f"Point(3) = {{{L!r}, {W!r}, 0, 1000}};", f"Point(4) = {{0, {W!r}, 0, 1000}};",
        "Line(1) = {1, 2};", "Line(2) = {2, 3};", "Line(3) = {4, 3};", "Line(4) = {1, 4};",
        "Curve Loop(1) = {1, 2, -3, -4};", "Plane Surface(100) = {1};",
        "Physical Surface(21) = {100};", "Physical Curve(22) = {4};", "Physical Curve(23) = {2};",
        "Physical Curve(24) = {1};", "Physical Curve(25) = {3};"]) + "\n"


def msh_node_count(path):
    """the number of nodes of a gmsh file (msh2: the line after $Nodes; msh4: its second number)"""
    with open(path) as f:
        for line in f:
            if line.startswith("$Nodes"):
                w = next(f).split()
                return int(w[0]) if len(w) == 1 else int(w[1])
    raise ValueError(f"{path}: no $Nodes section")


def empty_partitions(meshdir):
    """the part.*.nodes files of meshdir's partitioning that are empty"""
    import glob
    return [f for f in glob.glob(os.path.join(meshdir, "partitioning.*", "part.*.nodes")) if os.path.getsize(f) == 0]


def mesh_partitions(meshdir):
    """N of meshdir's partitioning.N (None if there is none or several)"""
    import glob
    ps = [d for d in glob.glob(os.path.join(meshdir, "partitioning.*")) if d.rsplit(".", 1)[-1].isdigit()]
    return int(ps[0].rsplit(".", 1)[-1]) if len(ps) == 1 else None


def remesh_spread(cfg, src: SavedState, name: str, src_mesh: str, n_procs: int, opt=None, run=None):
    """the new footprint `name` in the case dir (geo, size view, msh, Elmer mesh partitioned for n_procs) and src on
    it: (layout, x, frozen, diag, report lines)"""
    c = cfg["case"]
    opt = opt or default_opt(c)
    case = cfg["paths"]["case_dir"]
    L, W = float(c["spread_length_km"]) * 1e3, float(c["width"])
    ids, sx, sy, stri = read_footprint(os.path.join(case, src_mesh))
    nbs = len(ids)
    H = thickness(src, nbs)
    h, P, rep0 = size_field(sx, sy, stri, H, opt)
    rep = [rep0]
    write_pos(os.path.join(case, f"{name}_size.pos"), sx, sy, stri, h)
    with open(os.path.join(case, f"{name}.geo"), "w") as f:
        f.write(box_geo(L, W, f"{name}_size.pos"))
    gmsh, eg = c.get("gmsh", "gmsh"), c.get("elmergrid", "ElmerGrid")
    run = run or (lambda cmd: subprocess.run(cmd, cwd=case, check=True, stdout=subprocess.DEVNULL))
    run([gmsh, f"{name}.geo", "-1", "-2", "-format", "msh2", "-o", f"{name}.msh"])
    out = os.path.join(case, name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    # the number of partitions: at most n_procs, at least min_nodes_per_rank footprint nodes each (a coarse mesh on
    # 512 ranks leaves partitions empty), halved further while any partition comes out empty
    nn = msh_node_count(os.path.join(case, f"{name}.msh"))
    np_ = max(1, min(n_procs, nn // max(opt.get("min_nodes_per_rank", 40), 1)))
    while True:
        if os.path.isdir(out):
            shutil.rmtree(out)
        cmd = [eg, "14", "2", f"{name}.msh", "-autoclean", "-out", name]
        if np_ > 1:
            cmd += ["-metiskway", str(np_)]
        run(cmd)
        if np_ <= 1 or not empty_partitions(out):
            break
        log.warning(f"spread remesh: {name} on {np_} partitions leaves some empty: {np_ // 2}")
        np_ //= 2
    _, dx, dy, dtri = read_footprint(out)
    rep.append(f"new mesh {name}: {len(dx)} nodes, {len(dtri)} triangles, {np_} partitions (old {src_mesh}: {nbs} "
               f"nodes, {len(stri)} triangles)")
    layout, x, frozen, diag = interpolate_state(src, nbs, sx, sy, stri, dx, dy, along=lambda x_, y_: np.asarray(x_))
    for r in rep:
        log.info("spread remesh: " + r)
    return layout, x, frozen, diag, rep
