#!/usr/bin/env python3
"""
On tethys: the surface of the start state of the land strip runs (work/land_flowline_3d/state_078000.npz) as x, y, Zs of its surface nodes
-> work/land_flowline_3d/start_surface.npz, for plots/fssa_rel_rms.py (the reference's change since the START, so its first output
frame has a number to be divided by too).

    python3 plots/fssa_start_surface.py [STATE.npz] [OUT.npz]
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from ess.masks import _on
from ess.statefile import find_name, load_state

state = sys.argv[1] if len(sys.argv) > 1 else "work/land_flowline_3d/state_078000.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "work/land_flowline_3d/start_surface.npz"
s = load_state(state)
i = 0
for b in s.layout.blocks:
    n = len(b.gids)
    if b.name.lower() == "zs":
        g, zs = b.gids, s.x[i:i + n]
        break
    i += n
else:
    raise SystemExit(f"{state}: no Zs block")
x = _on(*s.diag[find_name(s.diag, "coordinate 1")], g)
y = _on(*s.diag[find_name(s.diag, "coordinate 2")], g)
np.savez(out, x=x, y=y, zs=zs)
print(f"{len(g)} surface nodes: x {x.min():g} .. {x.max():g}, y {y.min():g} .. {y.max():g}, Zs {zs.min():.2f} .. {zs.max():.2f} -> {out}")
