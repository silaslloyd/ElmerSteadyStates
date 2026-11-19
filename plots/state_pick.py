"""Sweep points given more than once (the same patch length d on several meshes, e.g. a coarse sweep's point and the
refined-margin sweep's, jobs/sweep_refined.py): only the one on the finest mesh (the most state values) is kept."""
import os

import numpy as np


def prefer_fine(paths):
    """paths with each sweep length d (meta::sweep_d, to 1 m) once: the state with the most values; other states
    (no sweep_d) kept as given. The order of first appearance is kept."""
    best, order, out = {}, [], []
    for p in dict.fromkeys(paths):
        with np.load(p) as z:
            d = float(z["meta::sweep_d"]) if "meta::sweep_d" in z.files else None
            n = int(z["x"].size)
        if d is None:
            out.append(p)
            continue
        k = round(d)
        if k not in best:
            order.append(k)
            best[k] = (n, p)
        elif n > best[k][0]:
            print(f"d = {d / 1e3:g} km: {os.path.relpath(p)} ({n} values) instead of {os.path.relpath(best[k][1])} "
                  f"({best[k][0]})")
            best[k] = (n, p)
    return out + [best[k][1] for k in order]
