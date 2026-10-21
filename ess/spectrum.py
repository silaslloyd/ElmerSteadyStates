"""
Spectrum of steady states with temperate beds of increasing size, at one parameter value (`ess spectrum`).

For k = 0, 1, 2, ... the frozen cold/temperate mask is temperate (T = T0) on the k glaciated bed columns
nearest the terminus and cold (flux condition) everywhere else. For each k:

  1. Newton at the fixed parameter, with that mask (no remasking: the bed may go above melting during the
     solve), starting from the previous k's state.
  2. Thermal consistency: a cold glaciated bed node above melting (T_b > T0 + tol) means the mask does not
     fit: the state is DISCARDED (recorded, not kept).
  3. Steady hydrology on the temperate patch (hydrology_steady below: a PLACEHOLDER until the model is
     specified). A water sheet thickness below zero at the cold/temperate boundary (the upstream end of the
     patch) means the patch cannot supply its own water there: DISCARDED.

Consistent states are kept: <work>/spectrum_<label>/points/state_XXXX.npz (XXXX = k; also viewable with
plots/plot_branch_states.py), and every k is a row of spectrum.csv with the checks. The run stops when the
patch reaches the divide, or at --k-max. Resubmitting resumes after the last k in spectrum.csv.
"""
from __future__ import annotations

import csv
import logging
import os

import numpy as np

from .masks import _get, _on, _xy_key
from .model import rms
from .statefile import find_name, save_state

log = logging.getLogger("ess")

T0 = 273.15


# ---------------------------------------------------------------- the bed columns and the masks
def bed_columns(layout, frozen, diag):
    """Glaciated bed nodes grouped into columns along x, ordered from the terminus inwards.
    Returns (gids of the mask nodes, column index per mask node (0 = at the terminus), column x)."""
    g_ct = frozen[find_name(frozen, "ColdTempMask")][0]
    gm = _on(*frozen[find_name(frozen, "GroundedMask")], g_ct, fill=0.0)
    xr, _, x = _xy_key(diag, g_ct)
    glac = gm > 0
    ux = np.unique(xr[glac])[::-1]                     # descending: terminus first
    col = np.full(len(g_ct), -1)
    pos = {v: i for i, v in enumerate(ux)}
    col[glac] = [pos[v] for v in xr[glac]]
    return g_ct, col, ux


def mask_for(k, g_ct, col):
    """temperate (+1) on columns 0..k-1 from the terminus, cold (-1) elsewhere"""
    return np.where((col >= 0) & (col < k), 1.0, -1.0)


# ---------------------------------------------------------------- consistency checks
def basal_temperature(layout, x, g):
    tn = next(b for b in layout.blocks if b.name.lower() == "temperature")
    return _on(tn.gids, layout.split(np.asarray(x))[tn.name], g)


def melt_rate(diag, g, rhoi, Lw):
    """Basal melt rate (m/a of ice) at temperate nodes from the heat solver's reaction loads, as
    ElmerStreams getmeltrate: m = -loads / (boundary weight * rho_i * L). Positive = melting."""
    L = _get(diag, "Temperature Loads")
    Wt = _get(diag, "Temperature Boundary Weights")
    if L is None or Wt is None:
        return None
    loads, w = _on(*L, g), _on(*Wt, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = -loads / (w * rhoi * Lw)
    return np.where(np.isfinite(m), m, np.nan)


def hydrology_steady(x_col, melt_col):
    """PLACEHOLDER for the steady hydrology on the temperate patch.

    x_col:    x of the temperate columns, from the cold/temperate boundary (index 0, upstream) to the
              terminus (last)
    melt_col: basal melt rate per column (m/a; < 0 = refreezing)
    returns:  water sheet thickness per column (same order), h[0] at the cold/temperate boundary

    Placeholder until the model is specified: h(x) = the net melt produced over the patch between x and
    the terminus, int_x^{x_term} m dx' (m^2/a, not a thickness). So h[0] < 0 means the patch refreezes more
    than it melts overall."""
    if len(x_col) == 0:
        return np.zeros(0)
    xs, ms = np.asarray(x_col, float), np.nan_to_num(np.asarray(melt_col, float))
    seg = 0.5 * (ms[1:] + ms[:-1]) * np.abs(np.diff(xs))       # trapezoids between columns
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])  # from each column to the terminus
    return tail


# ---------------------------------------------------------------- the loop
COLS = ["k", "n_temperate_cols", "x_boundary_km", "converged", "newton_its", "rms_F", "n_cold_above_Tm",
        "Tb_cold_max", "h_boundary", "melt_mean", "thermal_ok", "hydro_ok", "kept", "note"]


def run_spectrum(prob, saved, y0, p, newton_opt, outdir, *, k_max=None, k_step=1, tol_T=0.0, rhoi, Lw):
    from .newton import newton_krylov
    os.makedirs(os.path.join(outdir, "points"), exist_ok=True)
    csvf = os.path.join(outdir, "spectrum.csv")
    rows = []
    if os.path.exists(csvf):
        with open(csvf) as f:
            rows = list(csv.DictReader(f))
    layout, frozen = saved.layout, saved.frozen           # frozen: the model's dict (changed in place)
    diag0 = saved.diag
    g_ct, col, ux = bed_columns(layout, frozen, diag0)
    ncol = len(ux)
    kname = find_name(frozen, "ColdTempMask")
    k_end = ncol if k_max is None else min(k_max, ncol)
    y = y0
    k_start = 0
    if rows:
        k_start = int(rows[-1]["k"]) + k_step
        last = [r for r in rows if r.get("kept") == "1"]
        f_last = os.path.join(outdir, "points", f"state_{int(last[-1]['k']):04d}.npz") if last else None
        if f_last and os.path.exists(f_last):
            from .statefile import load_state
            y = prob.scale(load_state(f_last).x)
        log.info(f"spectrum: resuming at k = {k_start} ({len(rows)} rows in {csvf})")
    log.info(f"spectrum: p = {p:g}, {ncol} glaciated bed columns (terminus at x = {ux[0] / 1e3:.2f} km), "
             f"k = {k_start} .. {k_end} step {k_step}")

    for k in range(k_start, k_end + 1, k_step):
        ct = mask_for(k, g_ct, col)
        frozen[kname] = (g_ct, ct)
        r = newton_krylov(prob, y, p, newton_opt, label=f"spectrum k={k}")
        x = prob.unscale(r.y)
        diag = prob.last_diag
        Tb = basal_temperature(layout, x, g_ct)
        cold = (col >= 0) & (ct < 0)
        hot = cold & (Tb > T0 + tol_T)
        row = dict(k=k, n_temperate_cols=min(k, ncol), converged=int(r.converged), newton_its=r.iterations,
                   rms_F=r.history[-1] if r.history else np.nan, n_cold_above_Tm=int(hot.sum()),
                   Tb_cold_max=float(np.nanmax(Tb[cold])) if cold.any() else np.nan,
                   x_boundary_km=float(ux[k - 1] / 1e3) if k > 0 else np.nan)
        thermal_ok = r.converged and not hot.any()
        hydro_ok, h_b, mmean, note = True, np.nan, np.nan, ""
        if k > 0 and r.converged:
            m = melt_rate(diag, g_ct, rhoi, Lw)
            if m is None:
                hydro_ok, note = False, "no Temperature Loads / Boundary Weights in the diagnostics"
            else:
                mc = np.array([np.nanmean(m[col == c]) for c in range(k)])     # per column, terminus first
                xc = ux[:k]
                h = hydrology_steady(xc[::-1], mc[::-1])                       # boundary first
                h_b, mmean = float(h[0]), float(np.nanmean(mc))
                hydro_ok = bool(h_b >= 0)
        if not r.converged:
            note = "Newton did not converge"
        elif not thermal_ok:
            note = f"{int(hot.sum())} cold bed nodes above melting"
        elif not hydro_ok and not note:
            note = "water sheet < 0 at the cold/temperate boundary"
        kept = thermal_ok and hydro_ok
        row.update(h_boundary=h_b, melt_mean=mmean, thermal_ok=int(thermal_ok), hydro_ok=int(hydro_ok),
                   kept=int(kept), note=note)
        rows.append({c: row.get(c, "") for c in COLS})
        if kept:
            save_state(os.path.join(outdir, "points", f"state_{k:04d}.npz"), layout, x, p, dict(frozen), diag,
                       spectrum_k=k, x_boundary=row["x_boundary_km"] * 1e3 if k else np.nan)
        with open(csvf + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        os.replace(csvf + ".tmp", csvf)
        log.info(f"spectrum: k = {k}: {'KEPT' if kept else 'discarded'}"
                 + (f" ({note})" if note else "") + f"; boundary {row['x_boundary_km']:.2f} km, "
                 f"cold nodes above T0 {row['n_cold_above_Tm']}, h_boundary {h_b:.3g}")
        if r.converged:
            y = r.y                          # the next k starts from this state, kept or not
    return rows
