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


def is_sector(s: SavedState) -> bool:
    """a sector state (ess/case.py sector_geo: symmetric about the x-axis, so y < 0 on one half; strips have y >= 0)"""
    f = _all_fields(s)
    try:
        _, y = f[find_name(f, "coordinate 2")]
    except KeyError:
        return False
    return bool(np.min(y) < -1.0)


def radial(s: SavedState, divide: float | None = None) -> SavedState:
    """a copy of s whose "Coordinate 1" is the distance from the divide of a sector: the radius sqrt(x^2 + y^2), or
    divide - radius for an inverted sector (divide at the outer arc), so the transfer (by Coordinate 1) works on
    sectors as on strips"""
    from dataclasses import replace
    f = _all_fields(s)
    k1, k2 = find_name(f, "coordinate 1"), find_name(f, "coordinate 2")
    g1, x = f[k1]
    y = _lookup(f, k2, g1)
    new = (g1, (divide - np.hypot(x, y)) if divide else np.hypot(x, y))
    diag, frozen = dict(s.diag), dict(s.frozen)
    (diag if k1 in diag else frozen)[k1] = new
    return replace(s, diag=diag, frozen=frozen)


def with_bed_of(dst: SavedState, src: SavedState) -> SavedState:
    """dst with its bedrock replaced by src's bed as a function of Coordinate 1 (both radial for a sector: the bed of
    the strip at x = r), and Zb put on that bed where dst is grounded is left to transfer()"""
    from dataclasses import replace
    fs, fd = _all_fields(src), _all_fields(dst)
    gs, vs = src.frozen[find_name(src.frozen, "bedrock")]
    xs = np.round(_lookup(fs, "coordinate 1", gs), 3)
    ux, inv = np.unique(xs, return_inverse=True)
    prof = np.bincount(inv, vs) / np.bincount(inv)
    kd = find_name(dst.frozen, "bedrock")
    gd, _ = dst.frozen[kd]
    xd = _lookup(fd, "coordinate 1", gd)
    frozen = dict(dst.frozen)
    frozen[kd] = (gd, np.interp(xd, ux, prof))
    log.info(f"transfer: the bedrock of the target from the source's bed profile ({len(gd)} nodes)")
    return replace(dst, frozen=frozen)


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


def transfer_sector_pattern(src: SavedState, dst: SavedState, angle_deg: float, divide: float | None = None):
    """A STRIP state carried onto a SECTOR keeping its 2D pattern (ice streams, the cold/temperate bed): the sector
    node at distance s from the divide (r, or divide - r for an inverted sector) and arc fraction
    eta = (angle + A/2) / A in 0..1 takes the strip's column at x = s, y = ymin + eta W (the nearest one); every level
    of the extruded column is copied level for level (both meshes extruded with the same levels, ice and rock), so
    temperature, the cold/temperate mask, the grounded mask, Zs, Zb and the bed stay consistent with each other.
    The two periodic sides (eta = 0, 1) map to the strip's periodic sides (y = 0, W). Returns (x, frozen, diag) for
    dst (diag: dst's, with the strip's velocity and pressure for a warm start)."""
    from scipy.spatial import cKDTree
    fs, fd = _all_fields(src), _all_fields(dst)

    def cols(s_, f_):
        gz, _ = _block(s_, "Zs")
        nb = len(gz)
        g1, X = f_[find_name(f_, "coordinate 1")]
        Y = _lookup(f_, "coordinate 2", g1)
        o = np.argsort(g1)
        g1, X, Y = g1[o], X[o], Y[o]
        if not np.array_equal(g1[:nb], np.arange(1, nb + 1)):
            raise SystemExit("transfer: node ids are not 1.. per level (not an extruded footprint)")
        return nb, X[:nb], Y[:nb], len(g1)
    nbs, xs, ys, nns = cols(src, fs)
    nbd, xd, yd, nnd = cols(dst, fd)
    if nns // nbs != nnd // nbd or nns % nbs or nnd % nbd:
        raise SystemExit(f"transfer: different extruded levels ({nns // nbs} vs {nnd // nbd})")
    A = np.radians(float(angle_deg))
    r, ph = np.hypot(xd, yd), np.arctan2(yd, xd)
    sd = (divide - r) if divide else r
    eta = np.clip((ph + A / 2) / A, 0.0, 1.0)
    y0, W = float(ys.min()), float(np.ptp(ys))
    _, nc = cKDTree(np.c_[xs, ys]).query(np.c_[sd, y0 + eta * W])        # dst column -> nearest src column (0-based)
    log.info(f"transfer (pattern kept): {nbd} sector columns from {nbs} strip columns (strip {W / 1e3:.0f} km wide; "
             f"sector arc fraction -> y); distance from the divide {sd.min() / 1e3:.1f} .. {sd.max() / 1e3:.1f} km, "
             f"strip x {xs.min() / 1e3:.1f} .. {xs.max() / 1e3:.1f} km")

    def src_of(gd):                                   # dst gids -> the src gids at the same level of the mapped column
        lvl, c = (gd - 1) // nbd, (gd - 1) % nbd
        return lvl * nbs + nc[c] + 1

    def copy(rec_s, gd, partial=False):
        gsrc, vsrc = rec_s
        full = np.full(nns + 1, np.nan)
        full[gsrc] = vsrc
        v = full[src_of(gd)]
        if np.isnan(v).any() and not partial:
            raise SystemExit("transfer: a mapped node without a source value")
        return v
    x_new = dst.x.copy()
    sl = dst.layout.slices()
    for b in dst.layout.blocks:
        x_new[sl[b.name]] = copy(_block(src, b.name), b.gids)
    frozen = dict(dst.frozen)
    for k, (gd, _) in dst.frozen.items():
        try:
            frozen[k] = (gd, copy(src.frozen[find_name(src.frozen, k)], gd))
        except KeyError:
            log.info(f"transfer: frozen {k} not in the source, the target's kept")
    # the warm start (velocity, pressure; jobs/scaling_tests.cmd START_STATE): the strip's, the velocity turned with the
    # pattern: the strip's x (from the divide) along the sector's direction away from the divide (+e_r, inverted -e_r)
    # and its y along e_angle
    diag = dict(dst.diag)
    gv = {}
    for nm in ("Velocity 1", "Velocity 2", "Velocity 3", "Pressure"):
        try:
            rec = src.diag[find_name(src.diag, nm)]
        except KeyError:
            continue
        gall = np.arange(1, nnd + 1)
        v = copy(rec, gall, partial=True)          # the velocity / pressure only where the source has it (the ice)
        ok = np.isfinite(v)
        gv[nm] = (gall[ok], v[ok])
    if "Velocity 1" in gv and "Velocity 2" in gv:
        gall = gv["Velocity 1"][0]
        if not np.array_equal(gall, gv["Velocity 2"][0]):
            raise SystemExit("transfer: Velocity 1 and 2 on different nodes")
        c = (gall - 1) % nbd
        ca, sa = np.cos(ph[c]), np.sin(ph[c])
        sgn = -1.0 if divide else 1.0
        u, v = sgn * gv["Velocity 1"][1], gv["Velocity 2"][1]       # along e_r, e_angle
        gv["Velocity 1"] = (gall, u * ca - v * sa)
        gv["Velocity 2"] = (gall, u * sa + v * ca)
    for nm, rec in gv.items():
        old = [k for k in diag if k.lower() == nm.lower()]
        for k in old:
            del diag[k]
        diag[nm] = rec
    if gv:
        log.info(f"transfer (pattern kept): warm start {', '.join(gv)} from the strip (velocity turned)")
    gm = frozen[find_name(frozen, "GroundedMask")][1] if any(k.lower() == "groundedmask" for k in frozen) else None
    ct = frozen[find_name(frozen, "ColdTempMask")][1] if any(k.lower() == "coldtempmask" for k in frozen) else None
    msg = ""
    if gm is not None:
        msg += f"{int((gm > -0.5).sum())} grounded of {len(gm)} base nodes"
    if ct is not None:
        msg += f", {int((ct > 0).sum())} temperate"
    log.info(f"transfer (pattern kept): {msg}")
    return x_new, frozen, diag
