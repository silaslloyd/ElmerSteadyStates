"""
The frozen masks and their consistency with a state.

The map keeps two masks fixed (they are switches, so moving them inside a run would make the map
non-smooth): the cold/temperate bed mask ColdTempMask (+1 temperate: T = T0; -1 cold: flux condition)
and, for the marine flowline, the grounded mask GroundedMask (1 grounded, -1 floating). After a Newton
solve, `update_masks` checks them against the converged state and moves them where they are wrong:

  cold/temperate (grounded or glaciated bed only):
    cold -> temperate   where the basal temperature exceeds T0
    temperate -> cold   where the bed refreezes (Temperature Loads > 0: heat has to be removed to hold T0)
  grounded/floating (marine), by flotation: H_f = rho_w/rho_i (sea level - bed), sea level 0
    floating -> grounded  where the ice is thicker than flotation, H > H_f + tol/rho (its base is then
                          below the bed: there is no Zb limiter); Zb is put on the bed, Zs kept
    grounded -> floating  where it is thinner, H < H_f - tol/rho; the base is put at flotation,
                          Zb = -rho_i/rho_w H, and the surface at Zs = Zb + H
    (one criterion both ways, with a margin, so a node cannot flip back and forth; at a steady state
    the floating base is close to hydrostatic, so "thicker than flotation" = "base below the bed")

Newton is then repeated with the new masks, until nothing changes ([masks] auto in the config).
"""
from __future__ import annotations

import logging

import numpy as np

from .statefile import Records, StateLayout, find_name

log = logging.getLogger("ess")

T0 = 273.15


def _get(rec: Records, name):
    try:
        return rec[find_name(rec, name)]
    except KeyError:
        return None


def _on(g_from, v_from, g_to, fill=np.nan):
    """values of (g_from, v_from) at the gids g_to (fill where missing)"""
    out = np.full(len(g_to), fill, float)
    if len(g_from) == 0:
        return out
    o = np.argsort(g_from)
    gs, vs = np.asarray(g_from)[o], np.asarray(v_from, float)[o]
    k = np.clip(np.searchsorted(gs, g_to), 0, len(gs) - 1)
    ok = gs[k] == g_to
    out[ok] = vs[k[ok]]
    return out


def _xy_key(coords: Records, gids, digits=2):
    cx, cy = _get(coords, "coordinate 1"), _get(coords, "coordinate 2")
    if cx is None or cy is None:
        raise KeyError("no node coordinates in this state (diagnostics 'coordinate 1/2')")
    x, y = _on(*cx, gids), _on(*cy, gids)
    return np.round(x, digits), np.round(y, digits), x


def grounded_of(gm, marine):
    """bed nodes under grounded (marine: GroundedMask >= 0) or glaciated (land: GroundedMask > 0) ice"""
    return gm > -0.5 if marine else gm > 0


def update_masks(layout: StateLayout, x, frozen: Records, diag: Records, *, marine=False, rho_ratio=None,
                 tol=0.1, apply=True, grounding=True, periodic_width=None):
    """Check the frozen masks against the state x (and its diagnostics: Temperature Loads, coordinates).
    Returns (x_new, frozen_new, report). With apply=False nothing changes and only the report is made.
    report: counts of nodes per switch, and summary numbers (n_cold, n_temperate, n_grounded, GL position).
    grounding=False: leave the grounded mask alone (contact runs: Elmer moves it), still report on it.
    periodic_width: periodic sides y = 0 / y = W: the cold/temperate decision of each y = 0 node is copied to its
    y = W copy (the refreezing test reads the heat loads, which Elmer folds onto the y = 0 side: there the total,
    on the y = W side only a one-sided part, so the two copies could disagree).
    The grounded zone is kept contiguous from the divide: columns are scanned along the flow and the
    grounding line is the first column that is (or becomes) afloat; everything seaward of it floats."""
    x = np.asarray(x, float).copy()
    frozen = dict(frozen)
    rep = {}
    blocks = layout.split(x)
    names = {b.name.lower(): b.name for b in layout.blocks}
    gids = {b.name: b.gids for b in layout.blocks}

    gmr = _get(frozen, "GroundedMask")
    ctr = _get(frozen, "ColdTempMask")
    if gmr is None or ctr is None:
        return x, frozen, rep
    g_gm, gm = gmr[0], gmr[1].copy()
    g_ct, ct = ctr[0], ctr[1].copy()
    gm_ct = _on(g_gm, gm, g_ct, fill=-2.0)

    # ---- grounding line (marine)
    newly_grounded = np.zeros(len(g_ct), bool)
    if marine:
        zbn, zsn = names.get("zb"), names.get("zs")
        bedr = _get(frozen, "bedrock")
        if zbn is None or zsn is None or bedr is None:
            raise KeyError("marine masks need Zb and Zs in the state and bedrock in the frozen fields")
        gZb, Zb = gids[zbn], blocks[zbn].copy()
        gZs, Zs = gids[zsn], blocks[zsn].copy()
        gm_b = _on(g_gm, gm, gZb, fill=1.0)
        bed_b = _on(*bedr, gZb)
        xb, yb, xpos = _xy_key(diag, gZb)
        xs, ys, _ = _xy_key(diag, gZs)
        top = {k: i for i, k in enumerate(zip(xs, ys))}
        js = np.array([top.get(k, -1) for k in zip(xb, yb)])
        if (js < 0).any():
            raise KeyError(f"{int((js < 0).sum())} base nodes have no surface node above them")
        H = Zs[js] - Zb
        floating = gm_b < -0.5
        Hf = -bed_b / rho_ratio                                  # flotation thickness (sea level 0)
        # columns along the flow (the flowline is one element wide: two nodes per column)
        ux, col = np.unique(xb, return_inverse=True)
        col = col.ravel()
        nc = len(ux)
        Hc = np.bincount(col, H, nc) / np.bincount(col, None, nc)
        Hfc = np.bincount(col, Hf, nc) / np.bincount(col, None, nc)
        flc = np.bincount(col, floating.astype(float), nc) > 0      # a column with a floating node
        th = tol / rho_ratio
        i = 0
        while i < nc and ((not flc[i] and Hc[i] >= Hfc[i] - th) or (flc[i] and Hc[i] > Hfc[i] + th)):
            i += 1
        grounded_new = (col < i)
        if not grounding:
            grounded_new = ~floating
        ground = floating & grounded_new
        float_ = ~floating & ~grounded_new
        rep["n_base_below_bed"] = int((floating & (Zb < bed_b - 1.0)).sum())
        rep["base_below_bed_max"] = float(np.max(bed_b[floating] - Zb[floating])) if floating.any() else 0.0
        rep["n_grounded"] = int((~floating).sum())
        rep["x_gl"] = float(xpos[~floating].max()) if (~floating).any() else np.nan
        rep["to_grounded"] = int(ground.sum())
        rep["to_floating"] = int(float_.sum())
        if ground.any() or float_.any():
            gm_new_b = np.where(ground, 1.0, np.where(float_, -1.0, gm_b))
            Zb_new = np.where(ground, bed_b, Zb)
            zf = -rho_ratio * H
            Zb_new = np.where(float_, np.maximum(zf, bed_b), Zb_new)
            Zs_new = Zs.copy()
            Zs_new[js[float_]] = Zb_new[float_] + H[float_]
            if apply:
                blocks[zbn][:] = Zb_new
                blocks[zsn][:] = Zs_new
                gm = np.where(np.isin(g_gm, gZb[ground]), 1.0, np.where(np.isin(g_gm, gZb[float_]), -1.0, gm))
                gm_ct = _on(g_gm, gm, g_ct, fill=-2.0)
                newly_grounded = np.isin(g_ct, gZb[ground])
            gl_new = xpos[gm_new_b > -0.5].max() if (gm_new_b > -0.5).any() else np.nan
            rep["x_gl_new"] = float(gl_new)

    # ---- cold / temperate bed
    tn = names.get("temperature")
    if tn is not None:
        Tb = _on(gids[tn], blocks[tn], g_ct)
        bed = grounded_of(gm_ct, marine)
        new = ct.copy()
        new[newly_grounded] = np.where(Tb[newly_grounded] >= T0 - 0.01, 1.0, -1.0)
        warm = (new < 0) & bed & (Tb > T0)
        new[warm] = 1.0
        L = _get(diag, "Temperature Loads")
        freeze = np.zeros(len(g_ct), bool)
        if L is not None:
            freeze = (new > 0) & bed & ~newly_grounded & (_on(*L, g_ct, fill=-1.0) > 0)
            new[freeze] = -1.0
        if periodic_width:
            cx, cy = _get(diag, "coordinate 1"), _get(diag, "coordinate 2")
            if cx is not None and cy is not None:
                X, Y = _on(*cx, g_ct), _on(*cy, g_ct)
                master = {round(float(x_), 0): v for x_, y_, v in zip(X, Y, new) if abs(y_) < 1.0}
                im = np.where(np.abs(Y - periodic_width) < 1.0)[0]
                for i in im:
                    v = master.get(round(float(X[i]), 0))
                    if v is not None:
                        new[i] = v
        rep["to_temperate"] = int(warm.sum())
        rep["to_cold"] = int(freeze.sum())
        cold, temp = (ct < 0) & bed, (ct > 0) & bed
        rep["n_cold"], rep["n_temperate"] = int(cold.sum()), int(temp.sum())
        rep["Tb_cold_max"] = float(np.nanmax(Tb[cold])) if cold.any() else np.nan
        if apply:
            ct = new
    if apply:
        frozen[find_name(frozen, "GroundedMask")] = (g_gm, gm)
        frozen[find_name(frozen, "ColdTempMask")] = (g_ct, ct)
    return x, frozen, rep            # (the blocks are views of x: Zb / Zs were changed in place)


def n_changes(rep):
    return sum(int(rep.get(k, 0)) for k in ("to_grounded", "to_floating", "to_temperate", "to_cold"))


def describe(rep):
    parts = []
    if "x_gl" in rep:
        parts.append(f"GL {rep['x_gl'] / 1e3:.2f} km" + (f" -> {rep['x_gl_new'] / 1e3:.2f} km" if "x_gl_new" in rep else "")
                     + f" ({rep.get('to_grounded', 0)} nodes grounded, {rep.get('to_floating', 0)} floated)"
                     + (f"; floating base up to {rep['base_below_bed_max']:.1f} m below the bed"
                        f" ({rep['n_base_below_bed']} nodes > 1 m)" if rep.get("base_below_bed_max", 0) > 0.1 else ""))
    if "to_temperate" in rep:
        parts.append(f"bed {rep.get('to_temperate', 0)} cold -> temperate, {rep.get('to_cold', 0)} temperate -> cold")
    return "; ".join(parts) or "no masks"
