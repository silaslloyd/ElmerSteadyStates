"""
Spectrum of steady states with temperate beds of increasing size, at one parameter value (`ess spectrum`).

For k = 0, 1, 2, ... the frozen cold/temperate mask is temperate (T = T0) on the k glaciated bed columns
nearest the terminus and cold (flux condition) everywhere else. For each k:

  1. Newton at the fixed parameter, with that mask (no remasking: the bed may go above melting during the
     solve), starting from the previous k's state.
  2. Thermal consistency: a cold glaciated bed node above melting (T_b > T0 + tol) means the mask does not
     fit: the state is DISCARDED (recorded, not kept).
  3. Steady hydrology on the temperate patch (hydrology_steady): the flux q = int m dx from the
     cold/temperate boundary, then the sheet h from the margin (h = h0) back upstream through
     q = K(h) d/dx(c_N/h - phi0). If h reaches zero inside the patch (or is < 0 at the boundary), the
     patch cannot carry its water: DISCARDED. [hydrology] in the config: K0, alpha, h0, c_N.

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


def hydrology_steady(x_col, melt_col, phi0_col, hp):
    """Steady water sheet on the temperate patch (flowline).

        dq/dx = m                                  (m: basal melt rate, from the temperature loads)
        q(x_b) = 0                                 (no water enters at the cold/temperate boundary x_b)
        q = K(h) d/dx( c_N / h - phi0 ),  K(h) = K0 h^alpha (alpha > 1)
        h = h0 at the margin (the terminus end of the patch)

    q is integrated from the boundary to the margin; the flux law is then an initial-value problem for h
    from the margin back upstream:

        dh/dx = -(h^2 / c_N) * ( q / (K0 h^alpha) + dphi0/dx )

    x_col, melt_col, phi0_col: per temperate column, from the boundary (index 0) to the margin (last).
    phi0 as ElmerStreams' Phi0 = -sigma_zz + rho_w g z_b (here the Stokes pressure at the bed for -sigma_zz).
    hp: dict with K0, alpha, h0, c_N.
    Returns (h per column (NaN beyond where the integration stopped), q per column, ok, status):
    status "dry": h reached 0 inside the patch (ok = False: discarded); "flood": h exceeded h_max (default
    1e3 m; the solution blows up), recorded but not discarded; "ok"."""
    from scipy.integrate import solve_ivp
    xs = np.asarray(x_col, float)
    ms = np.nan_to_num(np.asarray(melt_col, float))
    ph = np.asarray(phi0_col, float)
    n = len(xs)
    q = np.concatenate([[0.0], np.cumsum(0.5 * (ms[1:] + ms[:-1]) * np.diff(xs))]) if n > 1 else np.zeros(n)
    h = np.full(n, np.nan)
    h[-1] = hp["h0"]
    if n < 2:
        return h, q, True, "ok"
    K0, al, cN = hp["K0"], hp["alpha"], hp["c_N"]
    dphi = np.diff(ph) / np.diff(xs)                     # piecewise constant between columns

    hmax = hp.get("h_max", 1e3)
    hdry = 1e-6 * hp["h0"]                   # "zero" (the flux law is singular at h = 0)

    def rhs(x, y):
        hh = min(max(y[0], 0.5 * hdry), 10 * hmax)
        j = min(max(np.searchsorted(xs, x) - 1, 0), n - 2)
        qq = np.interp(x, xs, q)
        return [-(hh * hh / cN) * (qq / (K0 * hh ** al) + dphi[j])]

    def dry(x, y):                          # h reaches 0: the patch cannot carry its water
        return y[0] - hdry
    dry.terminal, dry.direction = True, -1

    def flood(x, y):                        # h grows without bound (blows up in finite distance)
        return y[0] - hmax
    flood.terminal, flood.direction = True, 1
    try:
        sol = solve_ivp(rhs, (xs[-1], xs[0]), [hp["h0"]], t_eval=xs[::-1], events=(dry, flood), method="LSODA",
                        rtol=1e-8, atol=1e-3 * hdry)
    except ValueError:                       # event root finding failed right at h -> 0
        return h, q, False, "dry"
    hv = np.full(n, np.nan)
    got = sol.y[0] if sol.y.size else np.zeros(0)
    hv[::-1][:len(got)] = got
    dried = len(sol.t_events[0]) > 0
    flooded = len(sol.t_events[1]) > 0
    ok = not dried and (np.nanmin(hv) >= 0 if np.isfinite(hv).any() else True)
    status = "dry" if dried else ("flood" if flooded else "ok")
    return hv, q, bool(ok), status


# ---------------------------------------------------------------- the loop
COLS = ["k", "n_temperate_cols", "x_boundary_km", "converged", "newton_its", "rms_F", "n_cold_above_Tm",
        "Tb_cold_max", "h_boundary", "h_min", "q_margin", "hydro_status", "melt_mean", "thermal_ok", "hydro_ok", "kept", "note"]


def phi0_bed(diag, g, rhow, gravity, layout, x):
    """ElmerStreams' Phi0 = -sigma_zz + rho_w g z_b at the bed nodes, with the Stokes pressure for -sigma_zz
    and the (mapped) bed coordinate for z_b. None without Pressure in the diagnostics."""
    P = _get(diag, "Pressure")
    Z = _get(diag, "coordinate 3")
    if P is None or Z is None:
        return None
    return _on(*P, g) + rhow * gravity * _on(*Z, g)


def run_spectrum(prob, saved, y0, p, newton_opt, outdir, *, k_max=None, k_step=1, tol_T=0.0, rhoi, Lw,
                 rhow, gravity, hydro):
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
            phi0 = phi0_bed(diag, g_ct, rhow, gravity, layout, x)
            if m is None or phi0 is None:
                hydro_ok, note = False, "no Temperature Loads / Boundary Weights / Pressure in the diagnostics"
            else:
                mc = np.array([np.nanmean(m[col == c]) for c in range(k)])     # per column, terminus first
                pc = np.array([np.nanmean(phi0[col == c]) for c in range(k)])
                xc = ux[:k]
                h, q, ok, hstat = hydrology_steady(xc[::-1], mc[::-1], pc[::-1], hydro)   # boundary first
                row["hydro_status"] = hstat
                h_b, mmean = float(h[0]), float(np.nanmean(mc))
                hydro_ok = ok and bool(h_b >= 0)
                row["h_min"] = float(np.nanmin(h)) if np.isfinite(h).any() else np.nan
                row["q_margin"] = float(q[-1])
                if not ok:
                    note = "water sheet reaches 0 inside the temperate patch"
                elif hstat == "flood":
                    note = f"water sheet exceeds h_max = {hydro.get('h_max', 1e3):g} m (kept)"
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
