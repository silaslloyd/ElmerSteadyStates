#!/usr/bin/env python3
"""The thermal side of a state file: where it came from (meta), the temperature over the whole mesh and on the bed
(the ColdTempMask nodes), grounded vs floating, and the frozen cold/temperate mask.

  python3 periodic_test/thermal_start.py STATE.npz [--ref START.npz] [--dT 0.15 --cap 1e6]
--ref: also the temperature change from START (same mesh): everywhere, on the grounded and floating bed
--dT / --cap: the subtemperate sliding law (TDepSliding: factor exp(-(T - T0)/dT), capped at --cap) to say how many
grounded bed nodes are in its temperature-dependent range (T0 - dT ln(cap) .. T0) and how many above T0."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import load_state, find_name
from ess.masks import _on

ap = argparse.ArgumentParser()
ap.add_argument("state")
ap.add_argument("--ref")
ap.add_argument("--dT", type=float, default=0.15)
ap.add_argument("--cap", type=float, default=1e6)
a = ap.parse_args()
T0 = 273.15
s = load_state(a.state)
print(f"{a.state}: p = {s.p:g}")
for k, v in s.meta.items():
    t = str(v)
    print(f"  meta {k}: {t[:150]}")
tb = next(b for b in s.layout.blocks if b.name.lower() == "temperature")
T = s.layout.split(s.x)[tb.name]
q = lambda v: " ".join(f"{x:.2f}" for x in np.percentile(v, [0, 5, 50, 95, 100]))
print(f"temperature, all {len(T)} nodes (K; min 5% median 95% max): {q(T)}")
kc = find_name(s.frozen, "ColdTempMask")
g, ct = s.frozen[kc]
Tb = _on(tb.gids, T, g)
try:
    gm = _on(*s.frozen[find_name(s.frozen, "GroundedMask")], g, fill=np.nan)
except KeyError:
    gm = np.ones(len(g))
gr = gm > -0.5
print(f"bed nodes {len(g)}: grounded {gr.sum()}, floating {(~gr).sum()}")
print(f"  ColdTempMask: cold (-1) {(ct < 0).sum()}, temperate (+1) {(ct > 0).sum()}"
      f" (grounded temperate: {(gr & (ct > 0)).sum()})")
if gr.any():
    d = Tb[gr] - T0
    Tw = -a.dT * np.log(a.cap)
    print(f"  grounded bed T - T0 (K; min 5% median 95% max): {q(d)}")
    print(f"  grounded bed nodes above T0: {(d > 0).sum()}, within the sliding law's T-dependent range "
          f"({Tw:.2f} .. 0 K): {((d > Tw) & (d <= 0)).sum()}, colder (sliding at the cap): {(d <= Tw).sum()}")
if (~gr).any():
    print(f"  floating bed T - T0 (K; min 5% median 95% max): {q(Tb[~gr] - T0)}")
if a.ref:
    r = load_state(a.ref)
    rb = next(b for b in r.layout.blocks if b.name.lower() == "temperature")
    dTall = T - _on(rb.gids, r.layout.split(r.x)[rb.name], tb.gids)
    print(f"change from {a.ref} (K; min 5% median 95% max): all {q(dTall[np.isfinite(dTall)])}")
    db = _on(tb.gids, dTall, g)
    for name, m in (("grounded bed", gr), ("floating bed", ~gr)):
        if m.any():
            print(f"  {name}: {q(db[m][np.isfinite(db[m])])}")
print(f"non-finite temperatures: {(~np.isfinite(T)).sum()}")
