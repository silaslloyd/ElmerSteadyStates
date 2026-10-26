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
def ice_thickness(layout, x, g, diag):
    """ice thickness above the bed nodes g: Zs (state) at the surface node of the same column minus the
    (mapped) bed coordinate"""
    zsb = next(b for b in layout.blocks if b.name.lower() == "zs")
    zs = layout.split(np.asarray(x))[zsb.name]
    xb, yb, _ = _xy_key(diag, g)
    xs, ys, _ = _xy_key(diag, zsb.gids)
    top = {k: i for i, k in enumerate(zip(xs, ys))}
    js = np.array([top.get(k, -1) for k in zip(xb, yb)])
    zb = _on(*_get(diag, "coordinate 3"), g)
    return np.where(js >= 0, zs[np.maximum(js, 0)] - zb, np.nan)


def bed_columns(layout, frozen, diag, x=None, edge_thickness=None):
    """Glaciated bed nodes grouped into columns along x, ordered from the terminus inwards.
    The glaciated bed is the frozen GroundedMask (> 0: the glaciated mask written by `ess setup` from the
    INITIAL geometry), or, with edge_thickness (m), the columns from the divide up to the first column whose
    ice (Zs - bed, from the state x) is not thicker than that: the terminus of this state.
    Returns (gids of the mask nodes, column index per mask node (0 = at the terminus), column x)."""
    g_ct = frozen[find_name(frozen, "ColdTempMask")][0]
    xr, _, xpos = _xy_key(diag, g_ct)
    if edge_thickness is None:
        gm = _on(*frozen[find_name(frozen, "GroundedMask")], g_ct, fill=0.0)
        glac = gm > 0
    else:
        H = ice_thickness(layout, x, g_ct, diag)
        uxa = np.unique(xr)                              # ascending from the divide
        Hc = np.array([np.nanmean(H[xr == v]) for v in uxa])
        thin = np.flatnonzero(~(Hc > edge_thickness))
        last = thin[0] if thin.size else len(uxa)        # first column that is not glaciated
        glac = np.isin(xr, uxa[:last])
    ux = np.unique(xr[glac])[::-1]                     # descending: terminus first
    col = np.full(len(g_ct), -1)
    pos = {v: i for i, v in enumerate(ux)}
    col[glac] = [pos[v] for v in xr[glac]]
    return g_ct, col, ux


def mask_for(k, g_ct, col, beyond_temperate=False):
    """temperate (+1) on columns 0..k-1 from the terminus, cold (-1) elsewhere; beyond_temperate: the bed
    beyond the terminus (col < 0) temperate too"""
    t = (col >= 0) & (col < k)
    if beyond_temperate:
        t = t | (col < 0)
    return np.where(t, 1.0, -1.0)


# ---------------------------------------------------------------- consistency checks
def basal_temperature(layout, x, g):
    tn = next(b for b in layout.blocks if b.name.lower() == "temperature")
    return _on(tn.gids, layout.split(np.asarray(x))[tn.name], g)


def melt_rate(diag, g, rhoi, Lw):
    """Basal melt rate (m/a of ice) at temperate nodes from the heat solver's reaction loads, as
    ElmerStreams getmeltrate: m = -loads / (boundary weight * rho_i * L), with the strain heating solver's
    boundary weights (the heat solver's, when strain heating is off). Positive = melting."""
    L = _get(diag, "Temperature Loads")
    Wt = _get(diag, "Strain Heating Boundary Weights")    # as ElmerStreams getmeltrate (finite everywhere)
    if Wt is None:                                         # strain heating off: the heat solver's weights
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
        q = K(h) d/dx( c_N / h - phi0 ),  K(h) = K0 h^alpha (alpha > 1); no minus sign: N ~ c_N/h decreases with h
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
        # min_step: where h collapses towards 0 (dh/dx ~ 1/h) LSODA would otherwise shrink its step to
        # ~1e-11 m and stall ("t + h = t") before the dry event fires; with a floor it stops (status -1):
        # dry when h was falling there, flooding when rising
        import warnings
        with warnings.catch_warnings():      # "repeated error test failures": the stall, handled below
            warnings.simplefilter("ignore", UserWarning)
            sol = solve_ivp(rhs, (xs[-1], xs[0]), [hp["h0"]], t_eval=xs[::-1], events=(dry, flood), method="LSODA",
                            rtol=1e-8, atol=1e-3 * hdry, min_step=hp.get("min_step", 1e-3))
    except ValueError:                       # event root finding failed right at h -> 0
        return h, q, False, "dry"
    hv = np.full(n, np.nan)
    got = sol.y[0] if sol.y.size else np.zeros(0)
    hv[::-1][:len(got)] = got
    dried = len(sol.t_events[0]) > 0
    flooded = len(sol.t_events[1]) > 0
    if sol.status == -1 and not (dried or flooded):     # the step floor was hit
        xl = sol.t[-1] if len(sol.t) else xs[-1]          # the last output point (or the margin)
        hl = got[-1] if len(got) else hp["h0"]
        dried = bool(rhs(xl, [hl])[0] > 0)                # going upstream (dx < 0), h falls where dh/dx > 0
        flooded = not dried
    ok = not dried and (np.nanmin(hv) >= 0 if np.isfinite(hv).any() else True)
    status = "dry" if dried else ("flood" if flooded else "ok")
    return hv, q, bool(ok), status


# ---------------------------------------------------------------- one state of a spectrum
def evaluate_patch(r, x, diag, k, layout, g_ct, col, ux, ct, ncol, *, tol_T, hydro, phys):
    """The checks of a state with a temperate patch of k columns from the terminus (mask ct): Newton
    result r, state x, diagnostics diag. Thermal: no cold glaciated bed node above T0 + tol_T; hydrology:
    the steady water sheet on the patch (patch_hydrology). Returns (spectrum.csv row, the water-sheet
    fields to save with the state, kept, hydro_ok, note, h at the boundary)."""
    Tb = basal_temperature(layout, x, g_ct)
    cold = (col >= 0) & (ct < 0)
    hot = cold & (Tb > T0 + tol_T)
    row = dict(k=k, n_temperate_cols=min(k, ncol), converged=int(r.converged), newton_its=r.iterations,
               rms_F=r.history[-1] if r.history else np.nan, n_cold_above_Tm=int(hot.sum()),
               Tb_cold_max=float(np.nanmax(Tb[cold])) if cold.any() else np.nan,
               x_boundary_km=float(ux[k - 1] / 1e3) if k > 0 else np.nan)
    thermal_ok = r.converged and not hot.any()
    hydro_ok, h_b, mmean, note = True, np.nan, np.nan, ""
    hyd = {}                             # water sheet, flux and melt rate on the bed nodes (saved with the state)
    if k > 0 and r.converged:
        hydro_ok, h_b, mmean, note, extra, hyd = patch_hydrology(diag, layout, x, g_ct, col, ux, k, hydro=hydro,
                                                                 **phys)
        row.update(extra)
    if not r.converged:
        note = "Newton did not converge"
    elif not thermal_ok:
        note = f"{int(hot.sum())} cold bed nodes above melting"
    elif not hydro_ok and not note:
        note = "water sheet < 0 at the cold/temperate boundary"
    kept = thermal_ok and hydro_ok
    row.update(h_boundary=h_b, melt_mean=mmean, thermal_ok=int(thermal_ok), hydro_ok=int(hydro_ok),
               kept=int(kept), note=note)
    return row, hyd, kept, hydro_ok, note, h_b


def write_rows(csvf, rows):
    with open(csvf + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    os.replace(csvf + ".tmp", csvf)


# ---------------------------------------------------------------- the loop
COLS = ["k", "n_temperate_cols", "x_boundary_km", "converged", "newton_its", "rms_F", "n_cold_above_Tm",
        "Tb_cold_max", "h_boundary", "h_min", "q_margin", "hydro_status", "n_melt_filled", "melt_mean", "thermal_ok", "hydro_ok", "kept", "note"]


def fill_columns(xc, vals):
    """mean over each column's nodes; a column without any finite value is interpolated (in x) from the
    nearest columns that have one (Elmer's boundary weights are non-finite at a few nodes)"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.array([np.nanmean(a) if np.isfinite(a).any() else np.nan for a in vals], float)
    ok = np.isfinite(v)
    if ok.any() and not ok.all():
        o = np.argsort(xc)
        xs, vs, oks = np.asarray(xc)[o], v[o], ok[o]
        vs[~oks] = np.interp(xs[~oks], xs[oks], vs[oks])
        v[o] = vs
    return v


def phi0_bed(diag, g, rhow, gravity, layout, x):
    """ElmerStreams' Phi0 = -sigma_zz + rho_w g z_b at the bed nodes, with the Stokes pressure for -sigma_zz
    and the (mapped) bed coordinate for z_b. None without Pressure in the diagnostics."""
    P = _get(diag, "Pressure")
    Z = _get(diag, "coordinate 3")
    if P is None or Z is None:
        return None
    return _on(*P, g) + rhow * gravity * _on(*Z, g)


def patch_hydrology(diag, layout, x, g_ct, col, ux, k, *, rhoi, Lw, rhow, gravity, hydro):
    """The steady water sheet on a temperate patch of k columns from the terminus (bed columns g_ct, col, ux
    from bed_columns). Returns (hydro_ok, h at the cold/temperate boundary, mean melt rate, note,
    extra csv columns, {Water Sheet Thickness, Water Flux, Melt Rate} on the bed nodes)."""
    hydro_ok, h_b, mmean, note, row, hyd = True, np.nan, np.nan, "", {}, {}
    m = melt_rate(diag, g_ct, rhoi, Lw)
    phi0 = phi0_bed(diag, g_ct, rhow, gravity, layout, x)
    if m is None or phi0 is None:
        return False, h_b, mmean, "no Temperature Loads / Boundary Weights / Pressure in the diagnostics", row, hyd
    xc = ux[:k]
    mc = fill_columns(xc, [m[col == c] for c in range(k)])           # per column, terminus first
    pc = fill_columns(xc, [phi0[col == c] for c in range(k)])
    row["n_melt_filled"] = int(sum(not np.isfinite(m[col == c]).any() for c in range(k)))
    h, q, ok, hstat = hydrology_steady(xc[::-1], mc[::-1], pc[::-1], hydro)   # boundary first
    row["hydro_status"] = hstat
    hc, qc = h[::-1], q[::-1]                                          # terminus first, by column
    inpatch = (col >= 0) & (col < k)
    hn = np.full(len(g_ct), np.nan)
    qn = np.full(len(g_ct), np.nan)
    hn[inpatch], qn[inpatch] = hc[col[inpatch]], qc[col[inpatch]]
    o = np.argsort(g_ct)
    hyd = {"Water Sheet Thickness": (g_ct[o], hn[o]), "Water Flux": (g_ct[o], qn[o]),
           "Melt Rate": (g_ct[o], np.where(inpatch, m, np.nan)[o])}
    h_b, mmean = float(h[0]), float(np.nanmean(mc))
    hydro_ok = ok and bool(h_b >= 0)
    row["h_min"] = float(np.nanmin(h)) if np.isfinite(h).any() else np.nan
    row["q_margin"] = float(q[-1])
    if not ok:
        note = "water sheet reaches 0 inside the temperate patch"
    elif hstat == "flood":
        note = f"water sheet exceeds h_max = {hydro.get('h_max', 1e3):g} m (kept)"
    return hydro_ok, h_b, mmean, note, row, hyd


def run_spectrum(prob, saved, y0, p, newton_opt, outdir, *, k_max=None, k_step=1, tol_T=0.0, rhoi, Lw,
                 rhow, gravity, hydro, edge_thickness=None, stop_after_dry=None):
    """... stop_after_dry N: stop after N consecutive states that fail (Newton not converged, or the hydrology). The k = 0
    (all-cold) Newton result is always saved as <outdir>/k0_state.npz (the start of the next parameter)."""
    from .newton import newton_krylov
    os.makedirs(os.path.join(outdir, "points"), exist_ok=True)
    csvf = os.path.join(outdir, "spectrum.csv")
    rows = []
    if os.path.exists(csvf):
        with open(csvf) as f:
            rows = list(csv.DictReader(f))
    layout, frozen = saved.layout, saved.frozen           # frozen: the model's dict (changed in place)
    diag0 = saved.diag
    g_ct, col, ux = bed_columns(layout, frozen, diag0, x=prob.unscale(y0), edge_thickness=edge_thickness)
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
        f0 = os.path.join(outdir, "points", "state_0000.npz")
        if edge_thickness is not None and os.path.exists(f0):       # the terminus of the k = 0 steady state
            from .statefile import load_state
            s0 = load_state(f0)
            g_ct, col, ux = bed_columns(layout, frozen, s0.diag, x=s0.x, edge_thickness=edge_thickness)
            ncol = len(ux)
            k_end = ncol if k_max is None else min(k_max, ncol)
        log.info(f"spectrum: resuming at k = {k_start} ({len(rows)} rows in {csvf})")
    log.info(f"spectrum: p = {p:g}, {ncol} glaciated bed columns (terminus at x = {ux[0] / 1e3:.2f} km, from "
             + ("the frozen GroundedMask" if edge_thickness is None else f"ice thicker than {edge_thickness:g} m")
             + "), "
             f"k = {k_start} .. {k_end} step {k_step}")

    # consecutive failures: Newton not converged, or the water sheet reaching 0 (hydrology); a converged state
    # that passes the hydrology resets the count (thermal failures alone do not count)
    def failed(conv, hyd_ok):
        return (not conv) or (not hyd_ok)
    n_dry = 0
    for r_ in reversed(rows):                 # resuming: the current run of failures
        if int(r_.get("k", 0)) > 0 and failed(str(r_.get("converged")) == "1", str(r_.get("hydro_ok")) != "0"):
            n_dry += 1
        else:
            break
    k = k_start
    while k <= k_end:
        if stop_after_dry and n_dry >= stop_after_dry:
            log.info(f"spectrum: {n_dry} consecutive failures (Newton or hydrology), stopping at k = {k - k_step}")
            break
        ct = mask_for(k, g_ct, col)
        frozen[kname] = (g_ct, ct)
        r = newton_krylov(prob, y, p, newton_opt, label=f"spectrum k={k}")
        x = prob.unscale(r.y)
        diag = prob.last_diag
        if k == 0 and edge_thickness is not None and r.converged:
            # the terminus of the all-cold STEADY state (the start state may not be steady: its margin moves)
            g_ct, col, ux = bed_columns(layout, frozen, diag, x=x, edge_thickness=edge_thickness)
            ncol = len(ux)
            k_end = ncol if k_max is None else min(k_max, ncol)
            ct = mask_for(0, g_ct, col)
            log.info(f"spectrum: terminus of the k = 0 steady state at x = {ux[0] / 1e3:.2f} km "
                     f"({ncol} glaciated columns); the patches start there")
        row, hyd, kept, hydro_ok, note, h_b = evaluate_patch(
            r, x, diag, k, layout, g_ct, col, ux, ct, ncol, tol_T=tol_T, hydro=hydro,
            phys=dict(rhoi=rhoi, Lw=Lw, rhow=rhow, gravity=gravity))
        rows.append({c: row.get(c, "") for c in COLS})
        if r.converged:                       # discarded ones too (heat / hydrology): to look at them
            save_state(os.path.join(outdir, "points", f"state_{k:04d}.npz"), layout, x, p, dict(frozen), {**diag, **hyd},
                       spectrum_k=k, x_boundary=row["x_boundary_km"] * 1e3 if k else np.nan, kept=int(kept))
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
        if k == 0 and r.converged:
            save_state(os.path.join(outdir, "k0_state.npz"), layout, x, p, dict(frozen), diag)
            import json
            with open(os.path.join(outdir, "spectrum_meta.json"), "w") as fm:    # for plots/plot_spectrum.py
                json.dump(dict(p=p, n_columns=int(ncol), x_terminus_km=float(ux[0] / 1e3),
                               edge_thickness=edge_thickness), fm)
        if k > 0:
            n_dry = n_dry + 1 if failed(r.converged, hydro_ok) else 0
        k += k_step
    return rows


# ---------------------------------------------------------------- downward from the fully temperate bed
def run_spectrum_down(prob, saved, y0, p, newton_opt, outdir, *, k_step=5, k_min=0, tol_T=0.0, rhoi, Lw, rhow,
                      gravity, hydro, edge_thickness=None):
    """From a (fully) temperate steady state at p: patches of k = n, n - k_step, ... columns from the terminus
    (the cold bed spreading from the divide towards the margin; the bed beyond the terminus stays temperate,
    as in the temperate sweep), each Newton-solved from the last converged
    state (Stokes and heat coupled). When Newton fails, the step is cut (k_step - 1, ..., 1) from the last
    converged state; after a success the full step is used again; when a step of 1 fails, the run stops.
    The terminus (columns) is that of the start state. Written as a spectrum folder (spectrum.csv,
    points/state_XXXX.npz with the water sheet, spectrum_meta.json) for plots/plot_spectrum.py."""
    from .newton import newton_krylov
    from .statefile import load_state
    os.makedirs(os.path.join(outdir, "points"), exist_ok=True)
    csvf = os.path.join(outdir, "spectrum.csv")
    rows = []
    if os.path.exists(csvf):
        with open(csvf) as f:
            rows = list(csv.DictReader(f))
    layout, frozen = saved.layout, saved.frozen
    kname = find_name(frozen, "ColdTempMask")
    g_ct, col, ux = bed_columns(layout, frozen, saved.diag, x=prob.unscale(y0), edge_thickness=edge_thickness)
    ncol = len(ux)
    import json
    with open(os.path.join(outdir, "spectrum_meta.json"), "w") as fm:
        json.dump(dict(p=p, n_columns=int(ncol), x_terminus_km=float(ux[0] / 1e3), edge_thickness=edge_thickness,
                       direction="down"), fm)
    phys = dict(rhoi=rhoi, Lw=Lw, rhow=rhow, gravity=gravity)
    y, k_last = y0, None                   # the last converged state and its k
    for r_ in rows:
        if str(r_.get("converged")) == "1":
            sf = os.path.join(outdir, "points", f"state_{int(r_['k']):04d}.npz")
            if os.path.exists(sf) and (k_last is None or int(r_["k"]) < k_last):
                k_last, y = int(r_["k"]), prob.scale(load_state(sf).x)
    log.info(f"spectrum-down: p = {p:g}, {ncol} glaciated columns (terminus {ux[0] / 1e3:.2f} km), "
             f"k from {ncol if k_last is None else k_last} down to {k_min}, step {k_step}"
             + (f" (resuming at k = {k_last})" if k_last is not None else ""))

    def solve(k, y_start):
        ct = mask_for(k, g_ct, col, beyond_temperate=True)      # beyond the margin: temperate, as at the start
        frozen[kname] = (g_ct, ct)
        r = newton_krylov(prob, y_start, p, newton_opt, label=f"spectrum-down k={k}")
        x = prob.unscale(r.y)
        diag = prob.last_diag
        row, hyd, kept, hydro_ok, note, h_b = evaluate_patch(r, x, diag, k, layout, g_ct, col, ux, ct, ncol,
                                                             tol_T=tol_T, hydro=hydro, phys=phys)
        rows[:] = [r_ for r_ in rows if int(r_["k"]) != k]           # a retried k replaces its earlier row
        rows.append({c: row.get(c, "") for c in COLS})
        rows.sort(key=lambda r_: -int(r_["k"]))
        if r.converged:
            save_state(os.path.join(outdir, "points", f"state_{k:04d}.npz"), layout, x, p, dict(frozen),
                       {**diag, **hyd}, spectrum_k=k, x_boundary=row["x_boundary_km"] * 1e3 if k else np.nan,
                       kept=int(kept))
        write_rows(csvf, rows)
        log.info(f"spectrum-down: k = {k}: " + ("NOT converged" if not r.converged else
                 ("KEPT" if kept else "discarded") + (f" ({note})" if note else ""))
                 + (f"; boundary {row['x_boundary_km']:.2f} km" if k else "")
                 + f"; cold nodes above T0 {row['n_cold_above_Tm']}, h_boundary {h_b:.3g}")
        return r

    if k_last is None:                      # the start: the whole bed temperate (as the start state)
        r = solve(ncol, y0)
        if r.converged:
            k_last, y = ncol, r.y
        else:                               # the start state is a converged temperate state already: go on
            log.warning("spectrum-down: Newton did not converge at k = n (the start mask); stepping down from "
                        "the start state anyway")
            k_last, y = ncol, y0
    while k_last > k_min:
        here = set()
        for s in range(k_step, 0, -1):
            k = max(k_last - s, k_min)
            if k in here:                    # clamped at k_min: already tried from this state
                continue
            here.add(k)
            r = solve(k, y)
            if r.converged:
                k_last, y = k, r.y
                break
            log.info(f"spectrum-down: k = {k} from k = {k_last} did not converge"
                     + (f"; trying a step of {s - 1}" if s > 1 else ""))
        else:
            log.warning(f"spectrum-down: no converged state below k = {k_last} (even a step of 1); stopping")
            break
    return rows
