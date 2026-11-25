"""
The grounding line as an unknown of the steady state: root find on its position.

`ess newton --stokes-only` solves F(y) = Phi_T(y) - y = 0 with the grounded mask FROZEN, so it returns the steady state
for the grounding line it is given, not the steady grounding line. On a nearly flat bed the grounding line is almost
neutral: the steady grounding line moves by ~26 km for a 1 % change in the flux (x h_f'/h_f = 0.49, the stability
threshold being 0.4), and the transient takes ~100 kyr to get there. So the position is found as the root of a scalar
function of the imposed position x_g, by a secant method on top of the (frozen-mask) Newton:

    r(x_g) = x_f - x_g,

x_f = the flotation point of the converged state: where the thickness H crosses the flotation thickness
h_f = -bed / (rho_i / rho_w) along the flow (interpolated between the two columns, so not limited to the column
spacing). x_g = the last grounded column. r = 0 is a grounding line that agrees with its own steady state. r > 0: the ice
is still grounded-capable seaward of x_g (the steady grounding line is further down flow), r < 0 the opposite.
(Floating columns thicker than h_f, whose base would lie below the bed, count as r > 0: n_base_below_bed.)

This module has the parts that do not need Elmer: the residual of a saved state, the state with the grounding line moved
to a chosen x (ess/masks.py update_masks, gl_x), and the secant step. jobs/gl_newton_gl.py is the driver.
"""
from __future__ import annotations

import numpy as np

from .masks import _get, _on, _xy_key, update_masks
from .statefile import SavedState, find_name


def columns(saved: SavedState, rho_ratio: float) -> dict:
    """per column along the flow (marine flowline): x, thickness H, flotation thickness Hf, floating flag (a node of the
    column is afloat), and how far the base of the floating nodes lies below the bed"""
    names = {b.name.lower(): b.name for b in saved.layout.blocks}
    blocks = saved.layout.split(saved.x)
    gids = {b.name: b.gids for b in saved.layout.blocks}
    zbn, zsn = names["zb"], names["zs"]
    Zb, Zs = blocks[zbn], blocks[zsn]
    gZb, gZs = gids[zbn], gids[zsn]
    g_gm, gm = _get(saved.frozen, "GroundedMask")
    bedr = _get(saved.frozen, "bedrock")
    if bedr is None:
        raise KeyError("no bedrock in the frozen fields")
    gm_b = _on(g_gm, gm, gZb, fill=1.0)
    bed_b = _on(*bedr, gZb)
    xb, yb, _ = _xy_key(saved.diag, gZb)
    xs, ys, _ = _xy_key(saved.diag, gZs)
    top = {k: i for i, k in enumerate(zip(xs, ys))}
    js = np.array([top.get(k, -1) for k in zip(xb, yb)])
    if (js < 0).any():
        raise KeyError(f"{int((js < 0).sum())} base nodes have no surface node above them")
    H = Zs[js] - Zb
    Hf = -bed_b / rho_ratio
    floating = gm_b < -0.5
    ux, col = np.unique(xb, return_inverse=True)
    col = col.ravel()
    nc = len(ux)
    cnt = np.bincount(col, None, nc)
    below = np.where(floating, np.maximum(bed_b - Zb, 0.0), 0.0)
    return dict(x=ux, H=np.bincount(col, H, nc) / cnt, Hf=np.bincount(col, Hf, nc) / cnt,
                floating=np.bincount(col, floating.astype(float), nc) > 0,
                below=np.bincount(col, below, nc) / cnt)


def residual_of_columns(c: dict) -> dict:
    """x_gl (last grounded column), x_f (flotation point), r = x_f - x_gl, in m"""
    x, H, Hf, fl = c["x"], c["H"], c["Hf"], c["floating"]
    dif = H - Hf
    gr = np.flatnonzero(~fl)
    x_gl = float(x[gr].max()) if len(gr) else float("nan")
    x_f = float("nan")
    for j in range(1, len(x)):
        if dif[j - 1] > 0 >= dif[j]:
            x_f = float(x[j - 1] + dif[j - 1] / (dif[j - 1] - dif[j]) * (x[j] - x[j - 1]))
            break
    out = dict(x_gl=x_gl, x_f=x_f, r=x_f - x_gl,
               n_base_below_bed=int((c["below"] > 1.0).sum()), base_below_bed_max=float(c["below"].max()))
    if len(gr):
        k = gr[np.argmax(x[gr])]
        out["H_last_grounded"], out["Hf_last_grounded"] = float(H[k]), float(Hf[k])
    return out


def residual(saved: SavedState, rho_ratio: float) -> dict:
    return residual_of_columns(columns(saved, rho_ratio))


def shift_gl(saved: SavedState, gl_x: float, rho_ratio: float):
    """(x_new, frozen_new, report): the state with the grounded mask moved to gl_x (m): the columns at or upstream of it
    grounded, those seaward afloat; Zb / Zs of the switched columns as ess/masks.py does for a moved mask; the
    cold/temperate mask unchanged ([masks] ct = false)"""
    kct = find_name(saved.frozen, "ColdTempMask")
    ct = saved.frozen[kct]
    xn, frozen, rep = update_masks(saved.layout, saved.x, saved.frozen, saved.diag, marine=True, rho_ratio=rho_ratio,
                                   grounding=True, gl_x=gl_x)
    if not rep:
        raise RuntimeError("the grounding line was not moved: the state has no GroundedMask / ColdTempMask in its frozen fields")
    frozen = dict(frozen)
    frozen[find_name(frozen, "ColdTempMask")] = ct
    return xn, frozen, rep


def next_x(history, x_min, x_max, step0=10e3, max_step=25e3, bracket_frac=0.1):
    """The next grounding line position to try. history = [(x_g, r), ...] (m, m), in the order they were solved.
    One point: step0 in the direction r points (r > 0: seaward). A sign change among the points: the secant inside the
    narrowest bracket, kept at least bracket_frac of its width from the ends. No sign change: the secant of the last
    two points (its slope is expected negative, dr/dx_g < 0; if it is not, the step doubles in the direction r points),
    at most max_step per step. Always inside [x_min, x_max]. Returns (x, why)."""
    (x1, r1) = history[-1]
    if r1 == 0:
        return x1, "r = 0"
    if len(history) == 1:
        x = x1 + np.sign(r1) * step0
        return float(np.clip(x, x_min, x_max)), f"first step {np.sign(r1) * step0 / 1e3:+.1f} km (r = {r1:+.1f} m)"
    best = None
    for i in range(len(history)):
        for j in range(i + 1, len(history)):
            (xa, ra), (xb, rb) = history[i], history[j]
            if ra * rb < 0 and (best is None or abs(xb - xa) < abs(best[1][0] - best[0][0])):
                best = (history[i], history[j])
    if best is not None:
        (xa, ra), (xb, rb) = best
        lo, hi = (xa, xb) if xa < xb else (xb, xa)
        x = xa - ra * (xb - xa) / (rb - ra)
        pad = bracket_frac * (hi - lo)
        x = float(np.clip(x, lo + pad, hi - pad))
        return float(np.clip(x, x_min, x_max)), f"secant inside the bracket [{lo / 1e3:.2f}, {hi / 1e3:.2f}] km"
    (x0, r0) = history[-2]
    slope = (r1 - r0) / (x1 - x0) if x1 != x0 else 0.0
    if not np.isfinite(slope) or slope >= 0:
        step = np.sign(r1) * min(2 * abs(x1 - x0), max_step)
        why = f"slope {slope:+.3g} not negative: step {step / 1e3:+.1f} km in the direction of r"
    else:
        step = float(np.clip(-r1 / slope, -max_step, max_step))
        why = f"secant, slope {slope:+.3g}, step {step / 1e3:+.1f} km"
    return float(np.clip(x1 + step, x_min, x_max)), why


def root_estimate(history):
    """the position of the root from the points so far: the linear interpolation / extrapolation of the two points
    nearest to r = 0 (None with fewer than two)"""
    if len(history) < 2:
        return None
    pts = sorted(history, key=lambda p: abs(p[1]))[:2]
    (xa, ra), (xb, rb) = pts
    if rb == ra:
        return None
    return float(xa - ra * (xb - xa) / (rb - ra))
