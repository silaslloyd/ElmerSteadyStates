"""
A predictor for the marine bed sweep: the converged state of patch length d_old, with the structure around its
cold/temperate margin moved downstream with the head, as the first guess for d_new = d_old - delta.

The head moves from h_old = x_gl - d_old to h_new = h_old + delta. Every target node at X takes the source's value at
psi(X) = X - s(X), s = delta * w(X): w = 1 on the core [h_old - core + delta, h_new + core] (the margin's surface
feature and the temperature around it arrive delta downstream), ramping (smoothstep) to 0 over `taper` upstream and
downstream of it (far from the margin, near the divide and towards the grounding line, nothing changes). psi is
monotone when taper > delta (upstream, the source is stretched by delta over the taper). On the same mesh:

  - Temperature: layer by layer (the extruded mesh's node levels, counted from the bottom of each column: the ice in
    the normalised height zeta, the bedrock on its depth), interpolated in x;
  - Zs: the ice thickness moved, H = Zs - Zb (Zs_new(X) = Zb(X) + H(psi(X))), so the surface follows the bed;
  - background (sigma > 0, default 15 km): only the local ANOMALY of each profile (minus its Gaussian smoothing over
    sigma) is moved, the large-scale profile stays where it is (moving it too, the upstream taper shifts the whole ice
    sheet's slope: e.g. +65 m at 215 km, 25 km shift); sigma = 0: the whole profile moved;
  - Zb (and every other block): unchanged (the shifted region is grounded: Zb is the bed there).
"""
from __future__ import annotations

import logging

import numpy as np

from .masks import _on
from .statefile import SavedState, find_name

log = logging.getLogger("ess")


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def shift_weight(X, h_old, delta, core, taper):
    """w(X) (target coordinates, m): 1 on [h_old - core + delta, h_new + core], smoothstep ramps of width taper"""
    a, b = h_old - core + delta, h_old + delta + core
    up = _smoothstep((X - (a - taper)) / taper)
    dn = 1.0 - _smoothstep((X - b) / taper)
    return np.minimum(up, dn)


def _profile(x, v):
    """(sorted unique x, mean v) of scattered (x, v): the nodes across the flowline pooled"""
    xr = np.round(x, 2)
    ux, inv = np.unique(xr, return_inverse=True)
    return ux, np.bincount(inv, v) / np.bincount(inv)


def _background(px, pv, sigma):
    """the profile (px, pv) smoothed with a Gaussian of width sigma (m) on a uniform grid; 0 if sigma <= 0"""
    if sigma <= 0:
        return np.zeros_like(pv)
    from scipy.ndimage import gaussian_filter1d
    dx = max(float(np.min(np.diff(px))) if len(px) > 1 else 1.0, 50.0)
    xu = np.arange(px[0], px[-1] + dx, dx)
    vu = gaussian_filter1d(np.interp(xu, px, pv), sigma / dx, mode="nearest")
    return np.interp(px, xu, vu)


def _moved(px, pv, X, psi, sigma, pv_prev=None, ratio=0.0):
    """the profile at the nodes X with its anomaly (pv - background) taken from psi(X); with pv_prev (the previous
    point's profile on the same px): the background extrapolated, bg + ratio (bg - bg_prev) (the hybrid predictor)"""
    bg = _background(px, pv, sigma)
    an = pv - bg
    bgn = bg if pv_prev is None else bg + ratio * (bg - _background(px, pv_prev, sigma))
    return np.interp(X, px, bgn) + np.interp(psi(X), px, an)


def shift_margin(s: SavedState, h_old, delta, core=10e3, taper=None, x_gl=None, sigma=15e3, x_prev=None, ratio=0.0):
    """The state vector of s with the margin structure moved delta (m) downstream (see the module). Returns
    (x_new, info). HYBRID with x_prev (the previous converged point's state vector on the same layout) and ratio
    ((d_next - d) / (d - d_prev)): the smooth background (sigma) extrapolated by secant, the anomaly about it moved."""
    if x_prev is not None and sigma <= 0:
        raise ValueError("the hybrid predictor needs a background (sigma > 0)")
    taper = float(taper) if taper is not None else max(2.0 * delta, 20e3)
    if taper <= delta:
        raise ValueError(f"taper {taper / 1e3:g} km must exceed the shift {delta / 1e3:g} km (psi monotone)")
    end = h_old + delta + core + taper
    if x_gl is not None and end > x_gl - 10e3:
        log.warning(f"predictor: the shifted region reaches {end / 1e3:.1f} km, within 10 km of the grounding line "
                    f"{x_gl / 1e3:.1f} km")
    allc = {**s.diag, **s.frozen}
    cx, cy, cz = (allc[find_name(allc, f"coordinate {c}")] for c in (1, 2, 3))
    lay = s.layout
    parts = lay.split(s.x)
    prev = lay.split(np.asarray(x_prev, float)) if x_prev is not None else None
    out = {k: v.copy() for k, v in parts.items()}
    blk = {b.name.lower(): b for b in lay.blocks}
    psi = lambda X: X - delta * shift_weight(X, h_old, delta, core, taper)
    info = {}
    # temperature: layer by layer
    bT = blk["temperature"]
    g = bT.gids
    X, Y, Z = _on(*cx, g), _on(*cy, g), _on(*cz, g)
    col = np.unique(np.c_[np.round(X, 2), np.round(Y, 2)], axis=0, return_inverse=True)[1].ravel()
    ncol = col.max() + 1
    counts = np.bincount(col, minlength=ncol)
    if counts.min() != counts.max():
        raise ValueError(f"predictor: columns of {counts.min()} .. {counts.max()} temperature nodes (not one extruded "
                         f"mesh)")
    o = np.lexsort((Z, col))                        # by column, then upwards
    level = np.empty(len(g), int)
    level[o] = np.tile(np.arange(counts[0]), ncol)
    T = parts[bT.name]
    Tn = T.copy()
    for k in range(counts[0]):
        m = level == k
        px, pv = _profile(X[m], T[m])
        pvp = _profile(X[m], prev[bT.name][m])[1] if prev is not None else None
        Tn[m] = _moved(px, pv, X[m], psi, sigma, pvp, ratio)
    out[bT.name] = Tn
    info["T_change_max"] = float(np.max(np.abs(Tn - T)))
    # Zs: the thickness above Zb moved
    bS, bB = blk["zs"], blk["zb"]
    xs = _on(*cx, bS.gids)
    xb = _on(*cx, bB.gids)
    pb = _profile(xb, parts[bB.name])
    zb_at_s = np.interp(xs, *pb)
    ph = _profile(xs, parts[bS.name] - zb_at_s)
    php = None
    if prev is not None:
        zbp_at_s = np.interp(xs, *_profile(xb, prev[bB.name]))
        php = _profile(xs, prev[bS.name] - zbp_at_s)[1]
    Zs_n = zb_at_s + _moved(ph[0], ph[1], xs, psi, sigma, php, ratio)
    info["Zs_change_max"] = float(np.max(np.abs(Zs_n - parts[bS.name])))
    out[bS.name] = Zs_n
    x_new = np.concatenate([out[b.name] for b in lay.blocks])
    info.update(sigma_km=sigma / 1e3, core_km=(h_old - core + delta) / 1e3, core_end_km=(h_old + delta + core) / 1e3, taper_km=taper / 1e3)
    log.info(f"predictor: margin structure moved {delta / 1e3:g} km downstream (head {h_old / 1e3:.2f} -> "
             f"{(h_old + delta) / 1e3:.2f} km): full shift over {info['core_km']:.1f} .. {info['core_end_km']:.1f} km, "
             f"tapers {taper / 1e3:g} km, " + (f"anomaly about a {sigma / 1e3:g} km background moved" if sigma > 0 else
             "whole profiles moved") + (f", background extrapolated (secant, ratio {ratio:g})" if prev is not None
                                        else "") + f"; max change T {info['T_change_max']:.3g} K, Zs {info['Zs_change_max']:.3g} m")
    return x_new, info


# ---------------------------------------------------------------- secant predictor
def secant(y1, d1, y2, d2, d_next):
    """the linear extrapolation in the patch length from two converged states (y1 at d1, y2 at d2, the same layout) to
    d_next: y2 + (y2 - y1) (d_next - d2) / (d2 - d1)"""
    if abs(d2 - d1) < 1.0:
        raise ValueError("secant: the two lengths coincide")
    return y2 + (y2 - y1) * ((d_next - d2) / (d2 - d1))


def on_layout(src: SavedState, dst: SavedState):
    """src's state vector on dst's layout: src.x when the layouts are the same, else carried over (ess/transfer.py:
    temperature column by column, Zs and Zb in x; dst's own frozen fields, i.e. its bed, kept)"""
    if len(src.x) == len(dst.x) and all(a.name == b.name and np.array_equal(a.gids, b.gids)
                                        for a, b in zip(src.layout.blocks, dst.layout.blocks)):
        return np.asarray(src.x, float)
    from .transfer import transfer
    x, _ = transfer(src, dst)
    log.info("predictor: the earlier state carried over onto this mesh (ess transfer)")
    return x


# ---------------------------------------------------------------- extrapolation at fixed x
def _saturating(d3, v3, d_next, fallback):
    """d3: the three states' d, nearest to d_next first; v3 (3, n) their values. v(tau) = v_o + A (1 - exp(-lam tau)) with
    tau = |d_o - d| (o: the farthest state): lam from the ratio of the two increments (bisection), the value at d_next.
    Where the increments do not decelerate (ratio >= the linear one), `fallback` (the linear fit); where they reverse
    (ratio <= 0) the nearest value."""
    vn, vm, vo = v3
    do, dm, dn_ = d3[2], d3[1], d3[0]
    a, b, ts = abs(dm - do) / 1e3, abs(dn_ - dm) / 1e3, abs(d_next - do) / 1e3
    inc = vm - vo
    ok = np.abs(inc) > 1e-12
    R = np.where(ok, (vn - vm) / np.where(ok, inc, 1.0), np.nan)
    lin = b / a
    dec = ok & (R > 0) & (R < lin * (1 - 1e-3))
    lo = np.full(R.shape, 1e-6 / (a + b))
    hi = np.full(R.shape, 60.0 / min(a, b))
    for _ in range(60):                                     # g(lam) = e^{-lam a} (1 - e^{-lam b}) / (1 - e^{-lam a}) falls with lam
        mid = np.sqrt(lo * hi)
        g = np.exp(-mid * a) * (-np.expm1(-mid * b)) / (-np.expm1(-mid * a))
        big = g > R
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    lam = np.sqrt(lo * hi)
    A = inc / (-np.expm1(-lam * a))
    sat = vo + A * (-np.expm1(-lam * ts))
    out = np.where(dec, sat, fallback)
    return np.where(ok & (R <= 0), vn, out)


def extrapolate_values(X, V, ds, heads, d_next, head_next, order=2, npts=None, join=False, margin=0.0, relative=False,
                       blend=None, saturate=False):
    """The core: values at the nodes X (n,) of n_s states, V (n_s, n) their values, ds / heads (n_s,) their patch
    lengths and heads (same unit as X), extrapolated in d to d_next at FIXED x (the states ordered by nearness in d:
    the latest first). New patch (X >= head_next): the latest state's value, unchanged.
    Upstream of the new head, every node with at least TWO cold values (the two nearest states, x < their heads) is
    extrapolated: linearly through those two; where a third state was cold there too, the quadratic through the three
    (and so on up to degree `order`), blended in over `blend` (default the spacing of the two latest heads) from the
    edge of the third's cold zone inwards: no steps where the number of cold values changes.
    margin: a state's values within margin (unit of X) of its own head are not used (the cold/temperate jump).
    Nodes with fewer than two cold values (the head has just passed them): the latest value, or, with join=True
    (a profile, X ascending), the straight line from the last extrapolated node to the latest state's value at the new
    head (no steps). relative=True: instead the difference to the latest profile falls linearly to 0 at the new head (a
    surface stays above the temperate one; a straight line in x would cut below it).
    saturate=True: the third state's fit is not the quadratic but a SATURATING exponential v = v_o + A (1 - exp(-lam tau))
    through the three values (tau the progress in d): the surface at a fixed x tends to a limit as the head moves on and
    is not carried past it; where the three values do not decelerate the linear fit stays.
    Returns (values (n,), degree used (n,), -1 where the latest value is kept or joined)."""
    X, V = np.asarray(X, float), np.asarray(V, float)
    ds, heads = np.asarray(ds, float), np.asarray(heads, float)
    near = np.argsort(np.abs(ds - d_next), kind="stable")
    Vn, dn, hn = V[near], ds[near], heads[near] - margin
    use = X < head_next
    out = Vn[0].copy()
    deg = np.full(len(X), -1, int)
    if len(ds) < 2:
        return out, deg
    if blend is None:
        blend = max(abs(hn[0] - hn[1]), 1e-9)
    t = (dn - d_next) / 1e3                                    # the value at d_next: the constant coefficient
    prev = None
    for k in range(1, min(order, len(ds) - 1) + 1):
        valid = use & (X < hn[k])                              # the k+1 nearest states all cold here
        if not valid.any():
            break
        if saturate and k == 2:
            fit = _saturating(dn[:3], Vn[:3][:, valid], d_next, np.polyfit(t[:2], Vn[:2][:, valid], 1)[-1])
        else:
            fit = np.polyfit(t[:k + 1], Vn[:k + 1][:, valid], k)[-1]
        if prev is None:
            out[valid] = fit
        else:
            w = _smoothstep((hn[k] - X[valid]) / blend)
            out[valid] = prev[valid] + w * (fit - prev[valid])
        deg[valid] = k
        prev = out.copy()
    if join:
        full = deg >= 1
        if full.any():
            ia = int(np.argmax(np.where(full, X, -np.inf)))
            xa, xb = X[ia], head_next
            rest = use & (X > xa)
            if rest.any() and xb > xa:
                vb = np.interp(xb, X, Vn[0])                  # the latest state's value at the new head (temperate side)
                if relative:
                    out[rest] = Vn[0][rest] + (out[ia] - Vn[0][ia]) * (xb - X[rest]) / (xb - xa)
                else:
                    out[rest] = out[ia] + (vb - out[ia]) * (X[rest] - xa) / (xb - xa)
    return out, deg


def extrapolate(s: SavedState, d_s, head_s, hist, d_next, head_next, order=2, npts=4):
    """The state s (patch length d_s, head at head_s) with Zs and the temperature extrapolated in d to d_next at FIXED
    x (extrapolate_values), from earlier converged points hist = [(d, state vector, head), ...] on the same layout
    (physical units). Other blocks (Zb, ...) unchanged. Returns (x_new, info)."""
    allc = {**s.diag, **s.frozen}
    cx = allc[find_name(allc, "coordinate 1")]
    lay = s.layout
    sl = lay.slices()
    blk = {b.name.lower(): b for b in lay.blocks}
    x_new = np.asarray(s.x, float).copy()
    ds = [float(d_s)] + [float(d) for d, _, _ in hist]
    heads = [float(head_s)] + [float(h) for _, _, h in hist]
    info = {}
    for key in ("temperature", "zs"):
        b = blk[key]
        X = _on(*cx, b.gids)
        V = np.array([np.asarray(s.x, float)[sl[b.name]]] + [np.asarray(x, float)[sl[b.name]] for _, x, _ in hist])
        out, deg = extrapolate_values(X, V, ds, heads, d_next, head_next, order, npts)
        info[key] = dict(max_change=float(np.max(np.abs(out - V[0]))),
                         by_degree={int(k): int((deg == k).sum()) for k in np.unique(deg)},
                         upstream_nodes=int((X < head_next).sum()), no_history=int(((X < head_next) & (deg < 0)).sum()))
        x_new[sl[b.name]] = out
    log.info(f"predictor: extrapolated in d at fixed x from {len(ds)} point(s) (d = "
             f"{', '.join(f'{d / 1e3:g}' for d in sorted(ds))} km) to d = {d_next / 1e3:g} km (head {head_next / 1e3:.2f} km); "
             + "; ".join(f"{k}: max change {v['max_change']:.3g}, nodes by degree {v['by_degree']}, {v['no_history']} of "
                         f"{v['upstream_nodes']} upstream nodes without cold history" for k, v in info.items()))
    return x_new, info
