"""
Remeshing a CUT SECTOR footprint from a state (`ess remesh`): a new 2D footprint whose element size follows the
state, the same domain (the same jagged periodic sides, their segments only subdivided; the same arcs), partitioned,
and the state interpolated onto it level by level (Elmer extrudes the footprint: node id = level * nb + base id).

Element size h (m) at the old footprint's nodes, then limited in growth (h_i <= h_j + grade |x_i - x_j|):
  1. the base size: [case] res_points along the distance from the divide (as `ess setup`);
  2. coarse (h at least --coarse, default 2 km) on the grounded bed where it is frozen (ColdTempMask cold), or where
     the ice is fast and barely deforming (surface speed > --fast-speed and effective strain rate < --low-strain:
     the interior of an ice stream, plug flow);
  3. fine (h at most --fine, default 500 m) within --gl-band of the grounding line (either side: the full-Stokes
     strip beyond it and its migration) and in the shear margins (lateral shear strain rate |e_ns| > --margin-shear,
     n across the flow).
Strain rates from the surface velocity (the top level of the state's Velocity 1/2), per footprint triangle,
averaged to the nodes; the velocity in m/yr, so the rates in 1/yr.

The new geometry: ess/case.py's cut sector with a background size field (gmsh PostView) instead of size points; the
side chain the old one's (case/sector_side1.txt or the source mesh's own side file) with every segment longer than
the local size split into equal parts, side 2 its rotation (periodic curves). Partitioned like `ess setup`
(sector_partition, every periodic pair in one partition).

State interpolation: on every level, linear in the old footprint's triangles (a new node takes the old triangle it
lies in); the masks (GroundedMask, ColdTempMask) the nearest old node's; the bed radial when it is (by radius, no
smoothing over repeated remeshes), else linear; Zb on the bed where grounded and not below it elsewhere; the
coordinates the new mesh's (x, y) and Coordinate 3 interpolated.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

import numpy as np

from .statefile import Block, SavedState, StateLayout, find_name, load_state

log = logging.getLogger("ess")

MASKS = ("groundedmask", "coldtempmask")


# ------------------------------------------------------------------ footprints
def read_footprint(meshdir):
    """(ids, x, y, tri) of an Elmer 2D mesh directory: tri as 0-based rows into ids (ids sorted, 1..n)"""
    nodes = np.loadtxt(os.path.join(meshdir, "mesh.nodes"), ndmin=2)
    ids = nodes[:, 0].astype(np.int64)
    o = np.argsort(ids)
    ids, x, y = ids[o], nodes[o, 2], nodes[o, 3]
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise SystemExit(f"{meshdir}: node ids are not 1..n")
    tri = []
    for line in open(os.path.join(meshdir, "mesh.elements")):
        f = line.split()
        if f[2] == "303":
            tri.append([int(f[3]), int(f[4]), int(f[5])])
    if not tri:
        raise SystemExit(f"{meshdir}: no triangles (303) in mesh.elements")
    return ids, x, y, np.array(tri, np.int64) - 1


def node_areas(x, y, tri):
    a = 0.5 * np.abs((x[tri[:, 1]] - x[tri[:, 0]]) * (y[tri[:, 2]] - y[tri[:, 0]])
                     - (x[tri[:, 2]] - x[tri[:, 0]]) * (y[tri[:, 1]] - y[tri[:, 0]]))
    w = np.zeros(len(x))
    for c in range(3):
        np.add.at(w, tri[:, c], a / 3)
    return a, w


def node_gradient(x, y, tri, f):
    """d f / dx, d f / dy at the nodes: per triangle (linear), area-weighted average to the nodes"""
    xa, ya, fa = x[tri[:, 0]], y[tri[:, 0]], f[tri[:, 0]]
    dxb, dyb, dfb = x[tri[:, 1]] - xa, y[tri[:, 1]] - ya, f[tri[:, 1]] - fa
    dxc, dyc, dfc = x[tri[:, 2]] - xa, y[tri[:, 2]] - ya, f[tri[:, 2]] - fa
    det = dxb * dyc - dxc * dyb
    det = np.where(np.abs(det) > 0, det, 1.0)
    gx = (dfb * dyc - dfc * dyb) / det
    gy = (dfc * dxb - dfb * dxc) / det
    w = 0.5 * np.abs(det)
    G = np.zeros((len(x), 2))
    W = np.zeros(len(x))
    for c in range(3):
        np.add.at(G[:, 0], tri[:, c], w * gx)
        np.add.at(G[:, 1], tri[:, c], w * gy)
        np.add.at(W, tri[:, c], w)
    return G / np.maximum(W, 1e-300)[:, None]


class Locator:
    """the old footprint's triangle under each point: (tri index, barycentric weights); a point outside every
    triangle (round-off at the boundary) the nearest node's weight 1"""

    def __init__(self, x, y, tri):
        from scipy.spatial import cKDTree
        self.x, self.y, self.tri = x, y, tri
        self.cx, self.cy = x[tri].mean(1), y[tri].mean(1)
        self.tree = cKDTree(np.c_[self.cx, self.cy])
        self.ntree = cKDTree(np.c_[x, y])

    def __call__(self, px, py, k=12, quiet=False):
        """(nodes (n, 3), weights (n, 3)); a point in no triangle (just outside the old boundary, e.g. on an arc
        between the old arc's chords) takes the nearest candidate triangle's barycentric weights clamped to >= 0
        (the closest point of that triangle)"""
        n = len(px)
        T = np.full(n, -1)
        Wt = np.zeros((n, 3))
        bestT = np.zeros(n, int)
        bestM = np.full(n, -np.inf)
        bestW = np.zeros((n, 3))
        todo = np.arange(n)
        for kk in (min(k, len(self.tri)), min(96, len(self.tri))):      # then a wider search for the misses
            if not len(todo):
                break
            _, cand = self.tree.query(np.c_[px[todo], py[todo]], k=kk)
            cand = cand.reshape(len(todo), -1)
            sub = np.arange(len(todo))                 # rows of cand still looking
            for j in range(cand.shape[1]):
                if not len(sub):
                    break
                q = todo[sub]
                t = cand[sub, j]
                i0, i1, i2 = self.tri[t, 0], self.tri[t, 1], self.tri[t, 2]
                x0, y0 = self.x[i0], self.y[i0]
                d = (self.x[i1] - x0) * (self.y[i2] - y0) - (self.x[i2] - x0) * (self.y[i1] - y0)
                l1 = ((px[q] - x0) * (self.y[i2] - y0) - (self.x[i2] - x0) * (py[q] - y0)) / d
                l2 = ((self.x[i1] - x0) * (py[q] - y0) - (px[q] - x0) * (self.y[i1] - y0)) / d
                l0 = 1 - l1 - l2
                lm = np.minimum(np.minimum(l0, l1), l2)
                better = lm > bestM[q]
                bestM[q[better]] = lm[better]
                bestT[q[better]] = t[better]
                bestW[q[better]] = np.c_[l0, l1, l2][better]
                ok = lm > -1e-9
                T[q[ok]] = t[ok]
                Wt[q[ok]] = np.c_[l0, l1, l2][ok]
                sub = sub[~ok]
            todo = todo[sub]
        if len(todo):                                   # outside every triangle: the closest point of the best one
            T[todo] = bestT[todo]
            w = np.clip(bestW[todo], 0, None)
            Wt[todo] = w / np.maximum(w.sum(1), 1e-300)[:, None]
            if not quiet:
                tri = self.tri[T[todo]]
                qx, qy = np.sum(Wt[todo] * self.x[tri], 1), np.sum(Wt[todo] * self.y[tri], 1)
                dist = np.hypot(qx - px[todo], qy - py[todo])
                log.info(f"remesh: {len(todo)} new nodes just outside the old triangles (by at most {dist.max():.3g} m: "
                         f"the arcs between the old chords): the closest point's values")
        nodes = self.tri[np.maximum(T, 0)].copy()
        return nodes, np.clip(Wt, 0, 1) / np.maximum(np.clip(Wt, 0, 1).sum(1), 1e-300)[:, None]


# ------------------------------------------------------------------ the size field
def _level_values(rec, nb, level=None):
    """a record's values on one level of the extruded mesh, as an array over the footprint columns (NaN where
    missing); level None: the record's top level"""
    g, v = rec
    lv = (g - 1) // nb
    if level is None:
        level = int(lv.max())
    m = lv == level
    out = np.full(nb, np.nan)
    out[((g[m] - 1) % nb)] = v[m]
    return out, level


def base_values(rec, nb):
    """a base-only record (masks: one level, the ice base): its values per column"""
    g, v = rec
    lv = np.unique((g - 1) // nb)
    if len(lv) != 1:
        lv = [int(np.bincount(((g - 1) // nb)).argmax())]
    return _level_values(rec, nb, int(lv[0]))[0]


def grade(h, x, y, tri, g):
    """h_i <= h_j + g |x_i - x_j| along the mesh edges (repeated until nothing changes)"""
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
    e = np.unique(np.sort(e, 1), axis=0)
    L = np.hypot(x[e[:, 0]] - x[e[:, 1]], y[e[:, 0]] - y[e[:, 1]])
    h = h.copy()
    for it in range(10000):
        h0 = h.copy()
        np.minimum.at(h, e[:, 0], h[e[:, 1]] + g * L)
        np.minimum.at(h, e[:, 1], h[e[:, 0]] + g * L)
        if np.array_equal(h, h0):
            break
    return h


def size_field(src: SavedState, x, y, tri, h_res, opt):
    """the element size (m) at the old footprint's nodes and a report"""
    nb = len(x)
    f = {**src.diag, **src.frozen}
    gm = base_values(f[find_name(f, "groundedmask")], nb)
    ct = base_values(f[find_name(f, "coldtempmask")], nb)
    u, top = _level_values(f[find_name(f, "velocity 1")], nb)
    v, _ = _level_values(f[find_name(f, "velocity 2")], nb, top)
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    Gu, Gv = node_gradient(x, y, tri, u), node_gradient(x, y, tri, v)
    exx, eyy, exy = Gu[:, 0], Gv[:, 1], 0.5 * (Gu[:, 1] + Gv[:, 0])
    ee = np.sqrt(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2)
    sp = np.hypot(u, v)
    tx, ty = u / np.maximum(sp, 1e-9), v / np.maximum(sp, 1e-9)
    nx, ny = -ty, tx
    # e_ns = n . E . t (E symmetric): the lateral shear
    ens = nx * (exx * tx + exy * ty) + ny * (exy * tx + eyy * ty)
    grounded = np.nan_to_num(gm, nan=-1) > -0.5
    cold = np.nan_to_num(ct, nan=1) < 0
    h = h_res.copy()
    plug = grounded & (sp > opt["fast_speed"]) & (ee < opt["low_strain"])
    coarse = grounded & (cold | plug)
    h[coarse] = np.maximum(h[coarse], opt["coarse"])
    # the grounding line: grounded nodes with a floating neighbour (and the mask-0 nodes)
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
    gl = np.zeros(nb, bool)
    mix = grounded[e[:, 0]] != grounded[e[:, 1]]
    gl[e[mix].ravel()] = True
    gl |= np.abs(np.nan_to_num(gm, nan=-1)) < 0.5
    from scipy.spatial import cKDTree
    dgl = cKDTree(np.c_[x[gl], y[gl]]).query(np.c_[x, y])[0] if gl.any() else np.full(nb, np.inf)
    margin = np.abs(ens) > opt["margin_shear"]
    fine = (dgl < opt["gl_band"]) | (margin & (sp > 1.0))
    h[fine] = np.minimum(h[fine], opt["fine"])
    h = grade(h, x, y, tri, opt["grade"])
    _, area = node_areas(x, y, tri)
    est = lambda hh: float(np.sum(area / (0.433 * hh ** 2)))
    pc = lambda a: ", ".join(f"{q:g}%: {np.percentile(a, q):.3g}" for q in (50, 90, 99)) if len(a) else "-"
    rep = [f"surface speed on the grounded bed (m/yr): {pc(sp[grounded])}",
           f"effective strain rate there (1/yr): {pc(ee[grounded])}",
           f"lateral shear |e_ns| there (1/yr): {pc(np.abs(ens[grounded]))}",
           f"coarse (>= {opt['coarse']:g} m): cold {np.sum(area[grounded & cold]) / 1e6:.0f} km2, fast plug flow "
           f"{np.sum(area[plug & ~cold]) / 1e6:.0f} km2 (speed > {opt['fast_speed']:g} m/yr, strain rate < "
           f"{opt['low_strain']:g} /yr)",
           f"fine (<= {opt['fine']:g} m): grounding-line band {np.sum(area[dgl < opt['gl_band']]) / 1e6:.0f} km2 "
           f"(+- {opt['gl_band'] / 1e3:g} km, {int(gl.sum())} GL nodes), margins {np.sum(area[margin]) / 1e6:.0f} "
           f"km2 (|e_ns| > {opt['margin_shear']:g} /yr)",
           f"triangles: old mesh {len(tri)}, size field ~{est(h):.0f} (res_points alone ~{est(h_res):.0f})"]
    return h, rep


def _edges(tri):
    e = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
    return np.unique(np.sort(e, 1), axis=0)


def _smooth(f, e, n=2):
    """n passes of averaging each node with its edge neighbours"""
    for _ in range(n):
        s = f.copy()
        c = np.ones(len(f))
        np.add.at(s, e[:, 0], f[e[:, 1]])
        np.add.at(s, e[:, 1], f[e[:, 0]])
        np.add.at(c, e[:, 0], 1)
        np.add.at(c, e[:, 1], 1)
        f = s / c
    return f


def size_field_budget(src: SavedState, x, y, tri, h_res, opt):
    """the size from the distance to two curves, the background scaled to a triangle budget:

        h = min( h_min + grade max(d_GL - b_GL, 0),           the grounding line (+- b_GL at h_min)
                 h_ct + grade max(d_CT - b_CT, 0),            the cold/temperate transition on the grounded bed (the
                                                              stream margins; h_ct opt["fine_ct"], default h_min)
                 clip(s h_res, h_min, h_max) )                the background ([case] res_points), scaled by s
    and at least opt["coarse"] on a grounded bed away from both curves, opt["far_h"] farther than opt["far"] from both; s found by bisection so the estimated
    triangle count sum(area / (0.433 h^2)) is opt["max_triangles"] (no budget: s = 1). The stream margins of a
    thermally controlled bed are the cold/temperate transition, so the distance to it puts the resolution there
    without thresholds on strain rates. If even s at h_max overshoots the budget, the transition band is narrowed."""
    nb = len(x)
    f = {**src.diag, **src.frozen}
    gm = base_values(f[find_name(f, "groundedmask")], nb)
    ct = base_values(f[find_name(f, "coldtempmask")], nb)
    e = _edges(tri)
    grounded = np.nan_to_num(gm, nan=-1) > -0.5
    tempe = np.nan_to_num(ct, nan=1) > 0
    cold = grounded & ~tempe
    from scipy.spatial import cKDTree
    # the grounding line: grounded nodes with a floating neighbour (and the mask-0 nodes); the curve as edge midpoints
    mix = grounded[e[:, 0]] != grounded[e[:, 1]]
    glp = np.c_[0.5 * (x[e[mix, 0]] + x[e[mix, 1]]), 0.5 * (y[e[mix, 0]] + y[e[mix, 1]])]
    m0 = np.abs(np.nan_to_num(gm, nan=-1)) < 0.5
    glp = np.r_[glp, np.c_[x[m0], y[m0]]]
    # the cold/temperate transition: edges between a cold and a temperate node, both grounded
    tr = grounded[e[:, 0]] & grounded[e[:, 1]] & (tempe[e[:, 0]] != tempe[e[:, 1]])
    ctp = np.c_[0.5 * (x[e[tr, 0]] + x[e[tr, 1]]), 0.5 * (y[e[tr, 0]] + y[e[tr, 1]])]
    dist = lambda P: cKDTree(P).query(np.c_[x, y])[0] if len(P) else np.full(nb, np.inf)
    dgl, dct = dist(glp), dist(ctp)
    # directional bands (opt["prev_state"]: the state this mesh was made from, the curves at the last remesh): how far
    # each piece of each curve moved since, and to which side; the band ahead of a moving piece reaches look x that
    # distance (plus the minimum band), behind it and along still pieces only the minimum band
    band_gl = np.full(nb, float(opt["gl_band"]))
    band_ct = np.full(nb, float(opt["ct_band"]))
    prev = opt.get("prev_state")
    if prev is not None:
        fp = {**prev.diag, **prev.frozen}
        gm0 = base_values(fp[find_name(fp, "groundedmask")], nb)
        ct0 = base_values(fp[find_name(fp, "coldtempmask")], nb)
        if np.isnan(gm0).all():
            raise SystemExit("remesh --directional: the previous state is not on this mesh")
        g0 = np.nan_to_num(gm0, nan=-1) > -0.5
        t0 = np.nan_to_num(ct0, nan=1) > 0
        look, jump = opt.get("look", 2.5), opt.get("band_max", 50e3)
        hn = 2.0 * opt["fine"]                                 # displacements below this: noise of the masks
        report_dir = []

        def bands(Pnow, ein, pos_now, pos_prev, ok_prev, bmin, bdef, label):
            """per node the band of its nearest current curve piece: ahead / behind by the piece's motion"""
            if not len(Pnow):
                return np.full(nb, float(bdef))
            mix0 = ok_prev[e[:, 0]] & ok_prev[e[:, 1]] & (pos_prev[e[:, 0]] != pos_prev[e[:, 1]])
            P0 = np.c_[0.5 * (x[e[mix0, 0]] + x[e[mix0, 1]]), 0.5 * (y[e[mix0, 0]] + y[e[mix0, 1]])]
            if not len(P0):
                return np.full(nb, float(bdef))
            # each current piece (edge midpoint): how far the old curve is along the piece's normal (the curve's local
            # direction from the pieces within 3 km; the nearest old point's offset projected on the normal, so a
            # stream's sides that grew with its head count as still), on the old positive side or not
            d0, i0 = cKDTree(P0).query(Pnow)
            off = P0[i0] - Pnow
            tn = cKDTree(Pnow)
            nrm = np.zeros_like(Pnow)
            for q, nb_ in enumerate(tn.query_ball_point(Pnow, r=3e3)):
                Q = Pnow[nb_] - Pnow[nb_].mean(0)
                if len(nb_) < 3:
                    nrm[q] = off[q] / max(np.hypot(*off[q]), 1e-9)
                    continue
                w, v = np.linalg.eigh(Q.T @ Q)
                nrm[q] = v[:, 0]                                   # the smaller principal axis: the normal
            dn = np.abs(np.sum(off * nrm, 1))
            a, b = e[ein, 0], e[ein, 1]
            side0 = np.where(pos_prev[a] & pos_prev[b], 1.0, np.where(~pos_prev[a] & ~pos_prev[b], -1.0, 0.0))
            disp = np.maximum(dn - hn, 0.0) * side0                # signed: > 0 moved into the positive side
            new = d0 > jump                                        # no old curve near: a new piece, default band
            ahead_w = np.where(new, bdef, np.minimum(bmin + look * np.abs(disp), jump))
            j = cKDTree(Pnow).query(np.c_[x, y])[1]
            node_pos = np.where(pos_now, 1.0, -1.0)
            is_ahead = (np.sign(disp[j]) == node_pos) & (disp[j] != 0)
            bnd = np.where(new[j], bdef, np.where(is_ahead, ahead_w[j], bmin))
            mv = np.abs(disp) > 0
            report_dir.append(f"{label}: {mv.sum()} of {len(Pnow)} pieces moved (median {np.median(np.abs(disp[mv])) / 1e3 if mv.any() else 0:.1f} km, "
                              f"max {np.abs(disp).max() / 1e3:.1f} km; {int(np.sum(disp > 0))} into the "
                              f"{'grounded' if label == 'grounding line' else 'temperate'} side), {int(new.sum())} new")
            return bnd
        band_gl = bands(glp[:mix.sum()], mix, grounded, g0, np.ones(nb, bool), opt.get("band_min_gl", 5e3),
                        opt["gl_band"], "grounding line")
        band_ct = bands(ctp, tr, tempe, t0, g0 & grounded, opt.get("band_min_ct", 2e3), opt["ct_band"],
                        "cold/temperate transition")
        for r_ in report_dir:
            log.info(f"remesh: directional: {r_}")
    _, area = node_areas(x, y, tri)
    hmin, hmax, hc, g = opt["fine"], opt["h_max"], opt["coarse"], opt["grade"]
    count = lambda h: float(np.sum(area / (0.433 * h ** 2)))

    def build(sc, bct, graded=False):
        bct_n = band_ct * (bct / opt["ct_band"])               # the budget's narrowing scales the bands too
        hgl = hmin + g * np.maximum(dgl - band_gl, 0)
        hct = opt.get("fine_ct", hmin) + g * np.maximum(dct - bct_n, 0)   # the margins: their own (finer) size
        hbg = np.clip(sc * h_res, hmin, hmax)
        # the grounded bed away from both curves, cold (frozen) or temperate (the streams' interiors: fast, little
        # deformation), at least opt["coarse"]: only the boundaries are fine
        far = grounded & (dgl > band_gl) & (dct > bct_n)
        hbg[far] = np.maximum(hbg[far], min(hc, hmax))
        # and coarser still far from both (default 5 km beyond 200 km): the interior away from any margin
        far2 = grounded & (np.minimum(dgl, dct) > opt.get("far", np.inf))
        hbg[far2] = np.maximum(hbg[far2], min(opt.get("far_h", hc), hmax))
        h_ = np.minimum(np.minimum(hgl, hct), hbg)
        return grade(h_, x, y, tri, g) if graded else h_
    bct = opt["ct_band"]
    target = opt.get("max_triangles")
    # the budget search on the ungraded size (grading is the slow part), then once more with the target scaled by
    # the grading's effect on the count (grading only refines: the graded count is the larger)
    target0 = target
    if not target:
        sc, note = 1.0, "no budget: the background res_points as they are"
    else:
        target = float(target)
        sc_max = hmax / max(float(h_res.min()), 1.0)
        while count(build(sc_max, bct)) > target and bct > hmin:
            bct *= 0.5                                       # even the coarsest background overshoots: narrower
        if count(build(sc_max, bct)) > target:
            sc, note = sc_max, f"the budget {target:.0f} is below the floor {count(build(sc_max, bct)):.0f}: using it"
        elif count(build(1.0, bct)) <= target:              # the budget is a cap: never finer than res_points
            sc, note = 1.0, (f"the background res_points as they are: ~{count(build(1.0, bct, True)):.0f} triangles, "
                             f"under the budget")
        else:
            def search(tgt):
                lo, hi = 1.0, sc_max
                for _ in range(60):
                    mid = np.sqrt(lo * hi)
                    if count(build(mid, bct)) > tgt:
                        lo = mid
                    else:
                        hi = mid
                    if hi / lo < 1.005:
                        break
                return hi
            sc = search(target)
            ratio = count(build(sc, bct, True)) / max(count(build(sc, bct)), 1.0)
            sc = search(target / ratio)
            note = f"background res_points x {sc:.3g} (grading adds {100 * (ratio - 1):.1f}%)"
        if bct < opt["ct_band"]:
            note += f"; the transition band narrowed to +- {bct / 1e3:g} km to fit"
    h = build(sc, bct, True)
    bins = [0, 1.5 * hmin, 1000, 2000, 4000, 1e9]
    lab = [f"<= {1.5 * hmin:.0f} m", "..1 km", "1..2 km", "2..4 km", "> 4 km"]
    fr = [np.sum(area[(h > a) & (h <= b)]) / area.sum() * 100 for a, b in zip(bins[:-1], bins[1:])]
    rep = [f"curves: grounding line {len(glp)} points, cold/temperate transition {len(ctp)} points (grounded); "
           f"{note}",
           f"fine (<= {1.5 * hmin:.0f} m): GL band +- {opt['gl_band'] / 1e3:g} km {np.sum(area[dgl < opt['gl_band']]) / 1e6:.0f} km2, "
           f"transition band +- {bct / 1e3:g} km {np.sum(area[dct < bct]) / 1e6:.0f} km2; growth {g:g} m/m away from "
           f"them; grounded bed away from both (frozen or the streams' interiors) >= {min(hc, hmax):g} m",
           "area by size: " + ", ".join(f"{l} {v:.1f}%" for l, v in zip(lab, fr)),
           f"triangles: old mesh {len(tri)}, new ~{count(h):.0f}" + (f" (budget {target:.0f})" if target else "")
           + f"; footprint nodes ~{count(h) / 2:.0f}"]
    return h, rep


# ------------------------------------------------------------------ geometry
def refine_side(R, P, hfun):
    """the cut side chain (R, P: side 1, r increasing) with each segment longer than hfun at its midpoint split into
    equal straight parts (the side's shape unchanged); hfun(x, y) on arrays"""
    X, Y = R * np.cos(P), R * np.sin(P)
    ox, oy = [X[0]], [Y[0]]
    H = hfun(0.5 * (X[:-1] + X[1:]), 0.5 * (Y[:-1] + Y[1:]))
    for k in range(len(R) - 1):
        L = np.hypot(X[k + 1] - X[k], Y[k + 1] - Y[k])
        h = float(H[k])
        m = max(1, int(np.ceil(L / max(h, 1.0) - 0.2)))
        for j in range(1, m + 1):
            ox.append(X[k] + (X[k + 1] - X[k]) * j / m)
            oy.append(Y[k] + (Y[k + 1] - Y[k]) * j / m)
    ox, oy = np.array(ox), np.array(oy)
    return np.hypot(ox, oy), np.arctan2(oy, ox)


def write_pos(path, x, y, tri, h):
    with open(path, "w") as f:
        f.write('View "size" {\n')
        for a, b, c in tri:
            f.write(f"ST({float(x[a])!r},{float(y[a])!r},0,{float(x[b])!r},{float(y[b])!r},0,{float(x[c])!r},"
                    f"{float(y[c])!r},0){{{float(h[a])!r},{float(h[b])!r},{float(h[c])!r}}};\n")
        f.write("};\n")


def cut_geo_bg(R, P, A, divide_outer, pos_name):
    """ess/case.py _sector_cut_geo with this side chain and the size from the background view pos_name"""
    n = len(R)
    out = ["// generated by ess remesh: a cut sector, the size from a background field (the state), see ess/remesh.py",
           # Delaunay (5), as ess setup: random, isotropic triangles. Frontal-Delaunay (6) lays the triangles in rows
           # (straight lines of aligned edges, which the streams lock onto)
           "Mesh.Algorithm = 5;", "Mesh.MeshSizeFromPoints = 0;", "Mesh.MeshSizeExtendFromBoundary = 0;",
           "Mesh.MeshSizeFromCurvature = 0;", f'Merge "{pos_name}";',
           "Field[1] = PostView;", "Field[1].ViewIndex = 0;", "Background Field = 1;",
           "Point(100000) = {0, 0, 0, 1000};   // the centre"]
    for k in range(n):
        out.append(f"Point({k + 1}) = {{{float(R[k] * np.cos(P[k]))!r}, {float(R[k] * np.sin(P[k]))!r}, 0, 1000}};")
    for k in range(n):
        out.append(f"Point({n + k + 1}) = {{{float(R[k] * np.cos(P[k] + A))!r}, {float(R[k] * np.sin(P[k] + A))!r}, "
                   f"0, 1000}};")
    s1 = list(range(1, n))
    s2 = list(range(n, 2 * n - 1))
    for k in range(n - 1):
        out.append(f"Line({k + 1}) = {{{k + 1}, {k + 2}}};")
    for k in range(n - 1):
        out.append(f"Line({n + k}) = {{{n + k + 1}, {n + k + 2}}};")
    a_in, a_out = 2 * n - 1, 2 * n
    out.append(f"Circle({a_in}) = {{1, 100000, {n + 1}}};")
    out.append(f"Circle({a_out}) = {{{n}, 100000, {2 * n}}};")
    j = lambda v: ", ".join(str(t) for t in v)
    loop = s1 + [a_out] + [-l for l in reversed(s2)] + [-a_in]
    divide, front = ((a_out,), (a_in,)) if divide_outer else ((a_in,), (a_out,))
    out += [f"Transfinite Curve {{{j(s1 + s2)}}} = 2;     // one element edge per side segment",
            f"Curve Loop(1) = {{{j(loop)}}};", "Plane Surface(100) = {1};",
            f"Periodic Curve {{{j(s2)}}} = {{{j(s1)}}} Rotate {{{{0, 0, 1}}, {{0, 0, 0}}, {float(A)!r}}};",
            "Physical Surface(21) = {100};", f"Physical Curve(22) = {{{j(divide)}}};",
            f"Physical Curve(23) = {{{j(front)}}};", f"Physical Curve(24) = {{{j(s1)}}};",
            f"Physical Curve(25) = {{{j(s2)}}};"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ the state
def interpolate_state(src: SavedState, nbs, sx, sy, stri, dx, dy, divide=None):
    """src (on the old footprint sx, sy, stri; nbs columns) onto the new footprint dx, dy (same levels).
    Returns (layout, x, frozen, diag)."""
    nbd = len(dx)
    loc = Locator(sx, sy, stri)
    nodes, W = loc(dx, dy)
    near = nodes[np.arange(nbd), np.argmax(W, 1)]
    from scipy.spatial import cKDTree
    stree = cKDTree(np.c_[sx, sy])
    rs, rd = np.hypot(sx, sy), np.hypot(dx, dy)

    def onto(rec, name):
        g, v = rec
        lv, col = (g - 1) // nbs, (g - 1) % nbs
        levels = np.unique(lv)
        ismask = name.lower() in MASKS
        full = np.full((int(levels.max()) + 1, nbs), np.nan)
        full[lv, col] = v
        gids, vals = [], []
        whole = [L for L in levels if np.isfinite(full[L]).all()]
        if whole:                                          # complete levels: all at once
            F = full[whole]                                # (levels, nbs)
            val = F[:, near] if ismask else np.einsum("lnk,nk->ln", F[:, nodes], W)
            for i, L in enumerate(whole):
                gids.append(L * nbd + np.arange(nbd) + 1)
                vals.append(val[i])
        for L in [L for L in levels if L not in set(whole)]:
            have = np.isfinite(full[L])
            cover = np.ones(nbd, bool) if have.mean() > 0.99 else have[near]   # partial: where the nearest has it
            fill = full[L].copy()
            idx = np.where(have)[0]                        # gaps: the nearest old value
            _, k = cKDTree(np.c_[sx[idx], sy[idx]]).query(np.c_[sx[~have], sy[~have]])
            fill[~have] = full[L][idx[k]]
            val = fill[near] if ismask else np.sum(W * fill[nodes], 1)
            c = np.where(cover)[0]
            gids.append(L * nbd + c + 1)
            vals.append(val[c])
        g2, v2 = np.concatenate(gids).astype(np.int64), np.concatenate(vals)
        o = np.argsort(g2)
        return g2[o], v2[o]

    recs = {}
    for b in src.layout.blocks:
        recs[b.name] = onto((b.gids, src.layout.split(src.x)[b.name]), b.name)
    frozen = {k: onto(r, k) for k, r in src.frozen.items()}
    diag = {}
    for k, r in src.diag.items():            # only the warm start and the coordinates (the rest are outputs the next
        kl = k.lower()                       # run writes again)
        if kl in ("velocity 1", "velocity 2", "velocity 3", "pressure", "coordinate 3"):
            diag[k] = onto(r, k)
    # the water sheet: a state of the run (carried between runs as an output, "water sheet thickness" in diag), so
    # onto the new mesh too, as the frozen field the next run starts its sheet from; without it every remesh
    # restarted the sheet from its initial condition (jobs up to 31811)
    if not any(k.lower() == "water sheet thickness" for k in frozen):
        ks = [k for k in src.diag if k.lower() == "water sheet thickness"]
        if ks:
            frozen["Water Sheet Thickness"] = onto(src.diag[ks[0]], ks[0])
    # the coordinates: the new mesh's x, y on every level the old ones covered
    for k in [k for k in src.diag if k.lower() in ("coordinate 1", "coordinate 2")]:
        g, _ = src.diag[k]
        L = np.unique((g - 1) // nbs)
        gg = (L[:, None] * nbd + np.arange(nbd)[None, :] + 1).ravel()
        diag[k] = (gg, np.tile(dx if k.lower() == "coordinate 1" else dy, len(L)))
    # the bed: radial when it is (by radius: exact, no smoothing over repeated remeshes)
    kb = find_name(frozen, "bedrock")
    gS, bS = src.frozen[find_name(src.frozen, "bedrock")]
    lvS, colS = (gS - 1) // nbs, (gS - 1) % nbs
    m0 = lvS == lvS.min()
    bcol = np.full(nbs, np.nan)
    bcol[colS[m0]] = bS[m0]
    ok = np.isfinite(bcol)
    o = np.argsort(rs[ok])
    rr, bb = rs[ok][o], bcol[ok][o]
    radial = float(np.max(np.abs(np.interp(rs[ok], rr, bb) - bcol[ok]))) < 0.05 if ok.any() else False
    gB, vB = frozen[kb]
    if radial:
        vB = np.interp(rd[(gB - 1) % nbd], rr, bb)
        frozen[kb] = (gB, vB)
        log.info("remesh: the bed is radial: interpolated by radius")
    else:
        log.info("remesh: the bed is not radial: interpolated in the triangles")
    bed_col = np.full(nbd, np.nan)
    bed_col[(gB - 1) % nbd] = vB
    # Zb: on the bed where grounded, not below it elsewhere
    names = {b.name.lower(): b.name for b in src.layout.blocks}
    if "zb" in names:
        gm_g, gm_v = frozen[find_name(frozen, "groundedmask")]
        gmc = np.full(nbd, -1.0)
        gmc[(gm_g - 1) % nbd] = gm_v
        g, z = recs[names["zb"]]
        c = (g - 1) % nbd
        recs[names["zb"]] = (g, np.where(gmc[c] > -0.5, bed_col[c], np.maximum(z, bed_col[c])))
    layout = StateLayout([Block(b.name, recs[b.name][0], b.scale) for b in src.layout.blocks])
    x = np.concatenate([recs[b.name][1] for b in src.layout.blocks])
    return layout, x, frozen, diag


def remesh_distance(x, y, tri, fine, nlev):
    """'Remesh Distance' for elmer/RemeshTrigger.F90: at each footprint node the distance (m) to the nearest node whose
    local size (the mean length of its edges) is above 1.3 fine, on every one of nlev levels: (gids, values)"""
    from scipy.spatial import cKDTree
    e = _edges(tri)
    L = np.hypot(x[e[:, 0]] - x[e[:, 1]], y[e[:, 0]] - y[e[:, 1]])
    s, c = np.zeros(len(x)), np.zeros(len(x))
    for k in (0, 1):
        np.add.at(s, e[:, k], L)
        np.add.at(c, e[:, k], 1)
    h = s / np.maximum(c, 1)
    coarse = h > 1.3 * fine
    d = cKDTree(np.c_[x[coarse], y[coarse]]).query(np.c_[x, y])[0] if coarse.any() else np.full(len(x), 1e9)
    nb = len(x)
    g = (np.arange(nlev)[:, None] * nb + np.arange(nb)[None, :] + 1).ravel().astype(np.int64)
    return g, np.tile(d, nlev), h


def levels_of(src: SavedState, nb):
    """the number of extruded levels of a state on a footprint of nb nodes"""
    f = {**src.diag, **src.frozen}
    g, _ = f[find_name(f, "bedrock")]
    return int((int(g.max()) - 1) // nb + 1)


def mark(cfg, src: SavedState, mesh: str, fine: float):
    """'Remesh Distance' of a state on its own (old) mesh, so a run on it can trigger the first remesh"""
    ids, x, y, tri = read_footprint(os.path.join(cfg["paths"]["case_dir"], mesh))
    g, d, h = remesh_distance(x, y, tri, fine, levels_of(src, len(ids)))
    log.info(f"remesh: Remesh Distance on mesh {mesh}: {int(np.sum(h <= 1.3 * fine))} of {len(ids)} footprint nodes "
             f"fine (<= {1.3 * fine:g} m)")
    return g, d


# ------------------------------------------------------------------ driver
def remesh(cfg, src: SavedState, name: str, src_mesh: str, n_procs: int, opt: dict, run=None, dry=False):
    """the new footprint `name` in the case dir (geo, size view, msh, Elmer mesh, partitioning.<n_procs>) and the
    state on it. Returns (layout, x, frozen, diag, report lines)."""
    from .case import sector_partition
    c = cfg.get("case", {})
    case = cfg["paths"]["case_dir"]
    if not c.get("sector_angle") or c.get("sector_cut_seed") is None:
        raise SystemExit("remesh: only for cut sectors ([case] sector_angle and sector_cut_seed)")
    A = np.radians(float(c["sector_angle"]))
    divide = float(c["sector_divide_km"]) * 1e3 if c.get("sector_divide_km") else None
    front = float(c["front_km"]) * 1e3
    ids, sx, sy, stri = read_footprint(os.path.join(case, src_mesh))
    nbs = len(ids)
    # the base size (res_points along the distance from the divide)
    pts = np.array([[float(a) * 1e3, float(b)] for a, b in c["res_points"]])
    rs = np.hypot(sx, sy)
    sd = (divide - rs) if divide else rs
    h_res = np.interp(sd, pts[:, 0], pts[:, 1])
    if opt.get("mode", "curves") == "curves":
        if opt.get("prev") and opt.get("prev_state") is None:
            opt = dict(opt, prev_state=load_state(opt["prev"]))
            log.info(f"remesh: directional bands from the motion since {opt['prev']}")
        h, rep = size_field_budget(src, sx, sy, stri, h_res, opt)
    else:
        h, rep = size_field(src, sx, sy, stri, h_res, opt)
    for line in rep:
        log.info("remesh: " + line)
    if dry:
        return None, None, None, None, rep
    # the side: the old chain, its segments split to the new size (at both sides: min of side 1 and its image)
    side_src = os.path.join(case, f"sector_side1_{src_mesh}.txt")
    if not os.path.exists(side_src):
        side_src = os.path.join(case, "sector_side1.txt")
    R, P = np.loadtxt(side_src, ndmin=2).T
    hl = Locator(sx, sy, stri)

    def hfun(px, py):
        out = []
        for ang in (0.0, A):
            qx, qy = px * np.cos(ang) - py * np.sin(ang), px * np.sin(ang) + py * np.cos(ang)
            nd, w = hl(qx, qy, quiet=True)
            out.append(np.sum(w * h[nd], 1))
        return np.minimum(out[0], out[1])
    R2, P2 = refine_side(R, P, hfun)
    side_new = os.path.join(case, f"sector_side1_{name}.txt")
    np.savetxt(side_new, np.c_[R2, P2], fmt="%.17g", header=f"cut sector side 1 (r m, phi rad) of mesh {name}, "
               f"from {os.path.basename(side_src)} with its segments split ({len(R) - 1} -> {len(R2) - 1})")
    # a random factor on the size (log-uniform, +- opt["jitter"], one per old node, then graded): no regular patches
    # even where the size is uniform
    jit = float(opt.get("jitter", 0.15))
    if jit > 0:
        rng = np.random.default_rng(int(opt.get("seed", 0)) + len(R2))
        _, area_ = node_areas(sx, sy, stri)
        n0 = np.sum(area_ / h ** 2)
        hj = grade(h * np.exp(rng.uniform(-jit, jit, len(h))), sx, sy, stri, opt["grade"])
        # grading takes the smaller neighbour, so the jitter refines on balance: scaled back to the same count
        h = hj * np.sqrt(np.sum(area_ / hj ** 2) / n0)
    write_pos(os.path.join(case, f"{name}_size.pos"), sx, sy, stri, h)
    with open(os.path.join(case, f"{name}.geo"), "w") as f:
        f.write(cut_geo_bg(R2, P2, A, divide is not None, f"{name}_size.pos"))
    gmsh, eg = c.get("gmsh", "gmsh"), c.get("elmergrid", "ElmerGrid")
    run = run or (lambda cmd: subprocess.run(cmd, cwd=case, check=True))
    run([gmsh, f"{name}.geo", "-1", "-2", "-format", "msh2", "-o", f"{name}.msh"])
    out = os.path.join(case, name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    run([eg, "14", "2", f"{name}.msh", "-autoclean", "-out", name])
    _, dx, dy, dtri = read_footprint(out)
    rep.append(f"new mesh {name}: {len(dx)} nodes, {len(dtri)} triangles (old {nbs} nodes, {len(stri)} triangles); "
               f"side segments {len(R) - 1} -> {len(R2) - 1}")
    log.info("remesh: " + rep[-1])
    if n_procs > 1:
        sector_partition(case, name, n_procs, float(c["sector_angle"]), front, eg,
                         metisper=c.get("partitioner") == "metisper", msh=f"{name}.msh", side_file=side_new)
    layout, x, frozen, diag = interpolate_state(src, nbs, sx, sy, stri, dx, dy, divide)
    nlev = levels_of(src, nbs)
    g, d, hn = remesh_distance(dx, dy, dtri, opt["fine"], nlev)
    for k in [k for k in frozen if k.lower() == "remesh distance"]:
        del frozen[k]
    frozen["Remesh Distance"] = (g, d)
    rep.append(f"new mesh: {int(np.sum(hn <= 1.3 * opt['fine']))} of {len(dx)} footprint nodes fine (<= "
               f"{1.3 * opt['fine']:g} m)")
    return layout, x, frozen, diag, rep
