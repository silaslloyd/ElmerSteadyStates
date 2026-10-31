"""
Carry a converged state over to another mesh of the same flowline (e.g. from the full domain with a
regularised margin to the fixed-margin domain that ends at 0.8 L): `ess transfer`.

Both meshes are extruded columns over the same bed. Temperature is interpolated column by column:
in the ice on the normalised height zeta = (z - b) / (Zs - b), in the bedrock and filler on the depth
below the bed, then linearly in x between neighbouring columns. Zs is interpolated in x, and in
fixed-margin mode clamped to bed + edge_thickness at the end of the domain. The target mesh's own
frozen fields (bedrock, Zb, GroundedMask) are kept; the cold/temperate mask is taken from the source
(nearest bed node in x). Coordinates must be in both files (every run and `ess init` write them).

Marine states (Zb, the ice base, in the state vector; e.g. onto a mesh refined at the grounding line):
the ice is measured from the base Zb instead of the bed (zeta = (z - Zb) / (Zs - Zb), and below the base
the depth below Zb), Zb is interpolated in x and put on the bed where the ice is grounded, and the
grounded mask is taken from the source (nearest bed node in x) like the cold/temperate mask.
"""
from __future__ import annotations

import logging

import numpy as np

from .statefile import SavedState, find_name

log = logging.getLogger("ess")


def _lookup(rec, name, gids):
    g, v = rec[find_name(rec, name)]
    j = np.clip(np.searchsorted(g, gids), 0, len(g) - 1)
    if not np.array_equal(g[j], gids):
        raise KeyError(f"'{name}' does not cover all nodes")
    return v[j]


def _bed(f, gids, x):
    """bedrock elevation at the nodes (from the frozen field, or interpolated in x where it is missing)"""
    g, v = f[find_name(f, "bedrock")]
    j = np.clip(np.searchsorted(g, gids), 0, len(g) - 1)
    if np.array_equal(g[j], gids):
        return v[j]
    xb = _lookup(f, "coordinate 1", g)
    o = np.argsort(xb)
    return np.interp(x, xb[o], v[o])


def _block(s: SavedState, name):
    b = next(b for b in s.layout.blocks if b.name.lower() == name.lower())
    return b.gids, s.layout.split(s.x)[b.name]


def _all_fields(s: SavedState):
    return {**s.diag, **s.frozen}


def _profile(s: SavedState, name):
    """a state block (Zs, Zb) as a function of x (sorted, duplicates over the footprint width averaged)."""
    g, v = _block(s, name)
    x = _lookup(_all_fields(s), "coordinate 1", g)
    xr = np.round(x, 3)
    ux, inv = np.unique(xr, return_inverse=True)
    return ux, np.bincount(inv, v) / np.bincount(inv)


def _surface(s: SavedState):
    return _profile(s, "Zs")


def _has(s: SavedState, name):
    return any(b.name.lower() == name.lower() for b in s.layout.blocks)


def _base(s: SavedState, gids, x):
    """the ice base at the nodes: Zb (marine, in the state) interpolated in x, else the bed"""
    if _has(s, "Zb"):
        bx, bz = _profile(s, "Zb")
        return np.interp(x, bx, bz)
    return _bed(_all_fields(s), gids, x)


def _columns(s: SavedState):
    """Per column (unique x): sorted (zeta, T) in the ice and (depth below bed, T) in the rock."""
    gT, T = _block(s, "Temperature")
    f = _all_fields(s)
    x = _lookup(f, "coordinate 1", gT)
    z = _lookup(f, "coordinate 3", gT)
    b = _base(s, gT, x)
    sx, szs = _surface(s)
    zs = np.interp(x, sx, szs)
    xr = np.round(x, 3)
    # grouped by sorting (one pass): an unstructured footprint has a column per node, and a mask per column over all
    # nodes (as before) is O(columns x nodes), hours on a 3D triangle mesh
    ux, inv = np.unique(xr, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(ux) + 1))
    isice = z >= b - 1e-3
    zeta = np.clip((z - b) / np.maximum(zs - b, 1e-6), 0, 1)
    depth = b - z
    cols = {}
    for k, xc in enumerate(ux):
        m = order[bounds[k]:bounds[k + 1]]
        ice, rock = m[isice[m]], m[~isice[m]]
        zi = zeta[ice]
        o = np.argsort(zi)
        di = depth[rock]
        q = np.argsort(di)
        cols[xc] = (zi[o], T[ice][o], di[q], T[rock][q])
    return ux, cols


def _sample(xs, cols, x, ice, c):
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    x0, x1 = xs[i], xs[i + 1]
    w = float(np.clip((x - x0) / (x1 - x0), 0, 1)) if x1 > x0 else 0.0
    out = []
    for xc in (x0, x1):
        zi, Ti, di, Tr = cols[xc]
        if ice or len(di) == 0:
            out.append(np.interp(c, zi, Ti))
        else:
            out.append(np.interp(c, di, Tr))
    return (1 - w) * out[0] + w * out[1]


def transfer(src: SavedState, dst: SavedState, x_end_m: float | None = None, edge: float = 30.0):
    """Returns (x, frozen) for the layout of `dst`, from the physical state of `src`."""
    xs, cols = _columns(src)
    fd = _all_fields(dst)
    sl = dst.layout.slices()
    x_new = dst.x.copy()

    # temperature, on the target nodes as placed by the target's own (initial) surface
    gT, _ = _block(dst, "Temperature")
    x = _lookup(fd, "coordinate 1", gT)
    z = _lookup(fd, "coordinate 3", gT)
    b = _base(dst, gT, x)                     # the target's own (initial) base, which placed its nodes
    dx, dzs = _surface(dst)
    zs = np.interp(x, dx, dzs)
    T = np.empty(len(gT))
    for k in range(len(gT)):
        if z[k] >= b[k] - 1e-3:
            T[k] = _sample(xs, cols, x[k], True, np.clip((z[k] - b[k]) / max(zs[k] - b[k], 1e-6), 0, 1))
        else:
            T[k] = _sample(xs, cols, x[k], False, b[k] - z[k])
    tname = next(b_.name for b_ in dst.layout.blocks if b_.name.lower() == "temperature")
    x_new[sl[tname]] = T

    # surface
    gZ, _ = _block(dst, "Zs")
    xz = _lookup(fd, "coordinate 1", gZ)
    bz = _bed(fd, gZ, xz)
    sx, szs = _surface(src)
    Z = np.interp(xz, sx, szs)
    if x_end_m is not None:
        end = xz >= x_end_m - 1.0
        Z[end] = bz[end] + edge
        Z = np.maximum(Z, bz + edge)          # the regularised source can be thinner than the edge
        log.info(f"transfer: Zs clamped to bed + {edge:g} m at {int(end.sum())} end node(s)")
    zname = next(b_.name for b_ in dst.layout.blocks if b_.name.lower() == "zs")
    x_new[sl[zname]] = Z

    # frozen: the target's, with the cold/temperate mask (and, marine, the grounded mask) from the source
    # (nearest bed node in x)
    frozen = dict(dst.frozen)

    def nearest(name):
        ks, kd = find_name(src.frozen, name), find_name(dst.frozen, name)
        gs, vs = src.frozen[ks]
        xsrc = _lookup(_all_fields(src), "coordinate 1", gs)
        gd, _ = dst.frozen[kd]
        xdst = _lookup(fd, "coordinate 1", gd)
        o = np.argsort(xsrc)
        j = np.clip(np.searchsorted(xsrc[o], xdst), 0, len(o) - 1)
        jm = np.clip(j - 1, 0, len(o) - 1)
        near = np.where(np.abs(xsrc[o][jm] - xdst) < np.abs(xsrc[o][j] - xdst), jm, j)
        frozen[kd] = (gd, vs[o][near])
        return gd, xdst, vs[o][near]
    gd, _, ct = nearest("ColdTempMask")
    msg = f"temperate bed nodes {int((ct > 0).sum())} of {len(gd)}"

    if _has(dst, "Zb") and _has(src, "Zb"):   # marine: the ice base and the grounded mask
        gg, xg, gm = nearest("GroundedMask")
        gB, _ = _block(dst, "Zb")
        xB = _lookup(fd, "coordinate 1", gB)
        bB = _bed(fd, gB, xB)
        bx, bz = _profile(src, "Zb")
        Zb = np.interp(xB, bx, bz)
        jg = np.clip(np.searchsorted(gg, gB), 0, len(gg) - 1)
        grounded = (gg[jg] == gB) & (gm[jg] > -0.5)
        Zb = np.where(grounded, bB, np.maximum(Zb, bB))      # grounded: on the bed; floating: not below it
        zbname = next(b_.name for b_ in dst.layout.blocks if b_.name.lower() == "zb")
        x_new[sl[zbname]] = Zb
        x_gl = float(xg[gm > -0.5].max()) if (gm > -0.5).any() else np.nan
        msg += f"; grounding line at {x_gl / 1e3:.2f} km ({int((gm > -0.5).sum())} grounded bed nodes)"
    log.info(f"transfer: {len(gT)} temperatures, {len(gZ)} surface nodes; " + msg)
    return x_new, frozen
