#!/usr/bin/env python3
"""The sea level the floating ice implies, Zb + (rho_i/rho_w) H, in 5 km bins of radius, from a STATE (the Zs and Zb
variables themselves, not the mesh coordinates the VTU frames show; plots/float_check.py does the frames):

    python3 plots/float_check_state.py STATE.npz [SEA_LEVEL]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ess.statefile import find_name, load_state  # noqa: E402

s = load_state(sys.argv[1])
blocks = s.layout.split(s.x)
bl = {b.name.lower(): b for b in s.layout.blocks}
nb = min(len(bl["zs"].gids), len(bl["zb"].gids))
col = lambda g: (np.asarray(g) - 1) % nb
zs = np.full(nb, np.nan); zb = np.full(nb, np.nan)
zs[col(bl["zs"].gids)] = blocks[bl["zs"].name]
zb[col(bl["zb"].gids)] = blocks[bl["zb"].name]
f = {**s.diag, **s.frozen}
gm = np.full(nb, np.nan)
g, v = f[find_name(f, "groundedmask")]
gm[col(g)] = v
x = np.full(nb, np.nan); y = np.full(nb, np.nan)
for nm, arr in (("coordinate 1", x), ("coordinate 2", y)):
    g, v = f[find_name(f, nm)]
    arr[col(g)] = v
r = 920 / 1003.27
H = zs - zb
sl = zb + r * H
fl = (gm < -0.5) & np.isfinite(H)
R = np.hypot(x, y) / 1e3
print(f"{sys.argv[1]}: {nb} columns, floating {fl.sum()}; time {s.meta.get('time', '?')}"
      + (f"; expected sea level {sys.argv[2]}" if len(sys.argv) > 2 else ""))
for a in range(int(np.nanmin(R[fl])) // 5 * 5, int(np.nanmax(R[fl])) + 5, 5):
    m = fl & (R >= a) & (R < a + 5)
    if m.sum():
        print(f"r {a:4d}-{a+5:<4d} km: n {m.sum():5d}  H {np.median(H[m]):7.1f}  Zs {np.median(zs[m]):8.1f}  "
              f"Zb {np.median(zb[m]):8.1f}  implied sea level {np.median(sl[m]):8.1f} ({sl[m].min():.1f} .. {sl[m].max():.1f})")
