#!/usr/bin/env python3
"""The footprint nodes on the two periodic sides near a given x in a partitioning: which partitions list them, which
use them in an element, which are shared (part.N.shared), and the elements around them, to find out why one seam
column is not periodic.

  python3 periodic_test/seam_node.py MESHDIR/partitioning.N X_KM [--dx 0.3]"""
import argparse, glob, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pdir"); ap.add_argument("x", type=float); ap.add_argument("--dx", type=float, default=0.3)
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.pdir, "part.*.nodes")), key=lambda f: int(re.findall(r"part\.(\d+)\.nodes", f)[0]))
xy, listed, used, owner, shared, els = {}, {}, {}, {}, {}, {}
for f in files:
    p = int(re.findall(r"part\.(\d+)\.nodes", f)[0])
    for line in open(f):
        s = line.split(); g = int(s[0]); xy[g] = (float(s[2]), float(s[3])); listed.setdefault(g, set()).add(p)
    for line in open(f.replace(".nodes", ".elements")):
        s = line.split()
        for g in s[3:]:
            used.setdefault(int(g), set()).add(p)
            els.setdefault(int(g), []).append((p, int(s[0]), s[2], [int(q) for q in s[3:]]))
    sh = f.replace(".nodes", ".shared")
    if os.path.exists(sh):
        for line in open(sh):
            s = line.split(); shared.setdefault(int(s[0]), (p, [int(q) for q in s[2:]]))
ys = np.array([v[1] for v in xy.values()]); ymin, ymax = ys.min(), ys.max(); eps = 1e-5 * (ymax - ymin)
sel = sorted((g for g, (x, y) in xy.items() if abs(x / 1e3 - a.x) <= a.dx and (abs(y - ymin) < eps or abs(y - ymax) < eps)),
             key=lambda g: (xy[g][0], xy[g][1]))
for g in sel:
    x, y = xy[g]
    print(f"node {g}: x {x / 1e3:.3f} km, y {'0' if abs(y - ymin) < eps else 'W'}; listed in {sorted(listed[g])}, "
          f"used in {sorted(used.get(g, set()))}, shared entry {shared.get(g)}")
    for p, e, t, ns in els.get(g, []):
        print(f"     element {e} (type {t}) in partition {p}: nodes " +
              " ".join(f"{n}({xy[n][0] / 1e3:.2f},{xy[n][1] / 1e3:.2f})" for n in ns))
