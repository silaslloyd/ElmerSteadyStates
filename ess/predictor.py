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


def _moved(px, pv, X, psi, sigma):
    """the profile at the nodes X with its anomaly (pv - background) taken from psi(X)"""
    bg = _background(px, pv, sigma)
    an = pv - bg
    return np.interp(X, px, bg) + np.interp(psi(X), px, an)


def shift_margin(s: SavedState, h_old, delta, core=10e3, taper=None, x_gl=None, sigma=15e3):
    """The state vector of s with the margin structure moved delta (m) downstream (see the module). Returns
    (x_new, info)."""
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
        Tn[m] = _moved(px, pv, X[m], psi, sigma)
    out[bT.name] = Tn
    info["T_change_max"] = float(np.max(np.abs(Tn - T)))
    # Zs: the thickness above Zb moved
    bS, bB = blk["zs"], blk["zb"]
    xs = _on(*cx, bS.gids)
    xb = _on(*cx, bB.gids)
    pb = _profile(xb, parts[bB.name])
    zb_at_s = np.interp(xs, *pb)
    ph = _profile(xs, parts[bS.name] - zb_at_s)
    Zs_n = zb_at_s + _moved(ph[0], ph[1], xs, psi, sigma)
    info["Zs_change_max"] = float(np.max(np.abs(Zs_n - parts[bS.name])))
    out[bS.name] = Zs_n
    x_new = np.concatenate([out[b.name] for b in lay.blocks])
    info.update(sigma_km=sigma / 1e3, core_km=(h_old - core + delta) / 1e3, core_end_km=(h_old + delta + core) / 1e3, taper_km=taper / 1e3)
    log.info(f"predictor: margin structure moved {delta / 1e3:g} km downstream (head {h_old / 1e3:.2f} -> "
             f"{(h_old + delta) / 1e3:.2f} km): full shift over {info['core_km']:.1f} .. {info['core_end_km']:.1f} km, "
             f"tapers {taper / 1e3:g} km, " + (f"anomaly about a {sigma / 1e3:g} km background moved" if sigma > 0 else
             "whole profiles moved") + f"; max change T {info['T_change_max']:.3g} K, Zs {info['Zs_change_max']:.3g} m")
    return x_new, info
