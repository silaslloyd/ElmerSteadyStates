#!/usr/bin/env python3
"""Grounding line of raw Elmer extract files (work/*/cache/*.npz: keys g::NAME ids, v::NAME values).

python3 diag/cache_gl.py FILE [FILE ...]      one line per file: time of the file, GL (max x of grounded mask > -0.5), counts
Grounded = GroundedMask > -0.5 (the mask is defined on the bed nodes). x = Coordinate 1 of the same node ids.
"""
import os, sys, time
import numpy as np


def gl_of(path):
    z = np.load(path, allow_pickle=False)
    need = ("g::groundedmask", "v::groundedmask", "g::coordinate 1", "v::coordinate 1")
    if any(k not in z for k in need):
        return None
    gm_id = np.asarray(z["g::groundedmask"]).ravel()
    gm = np.asarray(z["v::groundedmask"]).ravel()
    cx_id = np.asarray(z["g::coordinate 1"]).ravel()
    cx = np.asarray(z["v::coordinate 1"]).ravel()
    order = np.argsort(cx_id)
    pos = np.searchsorted(cx_id[order], gm_id)
    pos = np.clip(pos, 0, len(cx_id) - 1)
    ok = cx_id[order][pos] == gm_id
    x = cx[order][pos][ok] / 1000.0
    m = gm[ok]
    g = m > -0.5
    if not g.any():
        return dict(gl=np.nan, ng=0, nf=int((~g).sum()))
    return dict(gl=float(x[g].max()), ng=int(g.sum()), nf=int((~g).sum()),
                xfl=float(x[~g].min()) if (~g).any() else np.nan)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        t = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
        try:
            r = gl_of(p)
        except Exception as e:
            r = None
            print(f"{t}  {os.path.basename(p)[:10]}  error {type(e).__name__}: {e}")
            continue
        if r is None:
            print(f"{t}  {os.path.basename(p)[:10]}  no groundedmask/coordinate keys")
        else:
            print(f"{t}  {os.path.basename(p)[:10]}  GL {r['gl']:9.2f} km   grounded {r['ng']:5d}  floating {r['nf']:5d}  first floating x {r.get('xfl', float('nan')):9.2f}")
