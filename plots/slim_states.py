#!/usr/bin/env python3
"""
Slim copies of chunk states for plots/plot_state_3d.py: only the top and bottom surfaces (Zs, Zb, the grounded
mask, the bedrock, the temperature, and the coordinates and velocities of those nodes), a few % of a 3D state.

    python plots/slim_states.py work/test_periodic_gl_50km  [OUT]
    python plots/slim_states.py STATE.npz [OUT.npz]        (one state; default STATE_slim.npz)
Copies every <folder>/<phase>_chunks/points/state_*.npz (and <phase>.npz) under the work dir to the same paths
under OUT (default <work dir>_slim). Files already slimmed (newer than their source) are skipped.
Run it on the cluster as a small job, e.g.  sbatch --ntasks=1 --time=01:00:00 --wrap "python3 plots/slim_states.py work/test_periodic_gl_50km"
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ess.statefile import StateLayout, find_name, load_state, save_state  # noqa: E402

KEEP_DIAG = ["coordinate 1", "coordinate 2", "coordinate 3", "Velocity 1", "Velocity 2", "Velocity 3", "GroundedMask"]
KEEP_FROZEN = ["GroundedMask", "bedrock"]


def slim(src, dst):
    s = load_state(src)
    blocks = s.layout.split(s.x)
    rec, names = {}, []
    for b in s.layout.blocks:
        if b.name.lower() in ("zs", "zb"):
            rec[b.name] = (b.gids, blocks[b.name])
            names.append(b.name)
    if not names:
        raise SystemExit(f"{src}: no Zs / Zb in the state")
    keep = np.unique(np.concatenate([rec[n][0] for n in names]))
    for b in s.layout.blocks:                       # the temperature on those nodes (the temperature panel)
        if b.name.lower() == "temperature":
            m = np.isin(b.gids, keep)
            rec[b.name] = (b.gids[m], blocks[b.name][m])
            names.append(b.name)

    def cut(records, wanted):
        out = {}
        for w in wanted:
            try:
                k = find_name(records, w)
            except KeyError:
                continue
            g, v = records[k]
            m = np.isin(g, keep)
            out[k] = (g[m], v[m])
        return out
    layout = StateLayout.from_records(rec, names, {n: 1.0 for n in names})
    x = np.concatenate([rec[b.name][1] for b in layout.blocks])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    save_state(dst, layout, x, s.p, cut(s.frozen, KEEP_FROZEN), cut(s.diag, KEEP_DIAG), **dict(s.meta))
    return os.path.getsize(src), os.path.getsize(dst)


def main():
    if sys.argv[1].endswith(".npz"):              # one state: STATE.npz [OUT.npz] (default STATE_slim.npz)
        d = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1][:-4] + "_slim.npz"
        sa, sb = slim(sys.argv[1], d)
        print(f"{sys.argv[1]}: {sa / 1e6:.0f} MB -> {d} {sb / 1e6:.1f} MB")
        return
    work = sys.argv[1].rstrip("/")
    out = sys.argv[2] if len(sys.argv) > 2 else work + "_slim"
    files = sorted(glob.glob(os.path.join(work, "*", "*_chunks", "points", "state_*.npz"))
                   + glob.glob(os.path.join(work, "*", "*.npz")))
    a = b = 0
    for f in files:
        d = os.path.join(out, os.path.relpath(f, work))
        if os.path.exists(d) and os.path.getmtime(d) >= os.path.getmtime(f):
            continue
        sa, sb = slim(f, d)
        a, b = a + sa, b + sb
        print(f"{os.path.relpath(f, work)}: {sa / 1e6:.0f} MB -> {sb / 1e6:.1f} MB", flush=True)
    print(f"{len(files)} states; written {a / 1e6:.0f} MB -> {b / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()
