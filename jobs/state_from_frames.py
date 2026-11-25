#!/usr/bin/env python3
"""
A state at an earlier time of a long run, rebuilt from its saved frames (plots/vtu_frames.py): the surface Zs and the
base Zb of one frame put onto a state of the same mesh (e.g. the run's start state). For a run that wrote no checkpoint:
the frames hold only the surface and the base, which is all a Stokes-only run needs (the temperature is frozen, the
velocity is solved first thing).

    python3 jobs/state_from_frames.py START.npz FRAMES.npz TIME_YR OUT.npz

START.npz: a state on the run's mesh (its Zs / Zb blocks are replaced, everything else is kept). FRAMES.npz: the
frames file; TIME_YR: the frame's time as printed by plots/margin_profile.py (the frames file's own times, which
include --t0), the nearest frame within 1 yr is used. The nodes are matched by (x, y): every Zs / Zb node of the
state must have exactly one frame node within 5 m (the frames are float32), else it stops.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.statefile import load_state, save_state  # noqa: E402


def block_xy(st, name):
    """(x, y) of the nodes of the state's block `name`, from the diagnostics 'coordinate 1' / 'coordinate 2'."""
    sl = st.layout.slices()[name]
    gids = next(b.gids for b in st.layout.blocks if b.name == name)
    xy = []
    for c in ("coordinate 1", "coordinate 2"):
        k = next((k for k in st.diag if k.lower() == c), None)
        if k is None:
            raise SystemExit(f"the state has no diagnostic {c!r}")
        g, v = st.diag[k]
        order = np.argsort(g)
        pos = np.searchsorted(g[order], gids)
        if (pos >= len(g)).any() or (g[order][np.minimum(pos, len(g) - 1)] != gids).any():
            raise SystemExit(f"{c}: not defined at every {name} node")
        xy.append(np.asarray(v)[order][pos])
    return sl, np.column_stack(xy)


def match(xy, fx, fy, what):
    d, i = cKDTree(np.column_stack([fx, fy]).astype(float)).query(xy)
    if d.max() > 5.0:
        raise SystemExit(f"{what}: a state node is {d.max():.1f} m from the nearest frame node (limit 5 m): other mesh?")
    if len(np.unique(i)) != len(i) or len(i) != len(fx):
        raise SystemExit(f"{what}: the state has {len(i)} nodes, the frame {len(fx)} ({len(np.unique(i))} matched once)")
    return i


def main():
    a = sys.argv[1:]
    if len(a) != 4:
        raise SystemExit(__doc__)
    start, frames, t, out = a[0], a[1], float(a[2]), a[3]
    fr = np.load(frames, allow_pickle=True)
    times = np.asarray(fr["frame_times"], float)
    k = int(np.argmin(np.abs(times - t)))
    if abs(times[k] - t) > 1.0:
        raise SystemExit(f"no frame at {t:g} yr; the frames are at {', '.join(f'{v:g}' for v in times)}")
    st = load_state(start)
    x = st.x.copy()
    for name, fx, fy, fz in (("Zs", fr["top_x"], fr["top_y"], fr["top_z"][k]),
                             ("Zb", fr["base_x"], fr["base_y"], fr["base_z"][k])):
        sl, xy = block_xy(st, name)
        i = match(xy, fx, fy, name)
        new = np.asarray(fz, float)[i]
        print(f"{name}: {sl.stop - sl.start} nodes, max change {np.abs(new - x[sl]).max():.2f} m, "
              f"median {np.median(np.abs(new - x[sl])):.2f} m; range {new.min():.1f} .. {new.max():.1f} m")
        x[sl] = new
    meta = {kk: v for kk, v in st.meta.items() if kk != "source"}
    save_state(out, st.layout, x, st.p, st.frozen, st.diag, source=f"{start} with Zs, Zb of frame t = {times[k]:g} yr of {frames}",
               **meta)
    print(f"frame t = {times[k]:g} yr of {frames} on {start} -> {out}")


if __name__ == "__main__":
    main()
