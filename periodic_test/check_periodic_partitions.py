#!/usr/bin/env python3
"""Check a partitioned Elmer mesh (partitioning.N/part.*.nodes|elements) with periodic sides in y (0 and the
maximum y): every node on the y = max side must be present in each partition where its counterpart on y = 0 is
needed, i.e. the partitions holding the slave node must also hold the master (what Elmer's periodic conditions
need). Also prints the balance (elements per partition) and the shared nodes.

  python3 periodic_test/check_periodic_partitions.py MESHDIR/partitioning.N  [--dim 2|3]
For an extruded 3D run the 2D footprint's partitioning is the one to check (Elmer extrudes it per partition)."""
import argparse, glob, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pdir")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.pdir, "part.*.nodes")), key=lambda f: int(re.findall(r"part\.(\d+)\.nodes", f)[0]))
xy = {}; where = {}; used = {}; nel = []; nshared = []; ncomp = []
for f in files:
    p = int(re.findall(r"part\.(\d+)\.nodes", f)[0])
    for line in open(f):
        s = line.split()
        g = int(s[0]); xy[g] = (float(s[2]), float(s[3]))
        where.setdefault(g, set()).add(p)
    ne = 0; par = {}
    def root(g):
        while par[g] != g:
            par[g] = par[par[g]]; g = par[g]
        return g
    for line in open(f.replace(".nodes", ".elements")):
        s = line.split(); ne += 1
        for g in s[3:]:
            used.setdefault(int(g), set()).add(p)
        # connected pieces of the partition (elements sharing a node)
        ns = [int(g) for g in s[3:]]
        for g in ns:
            par.setdefault(g, g)
        r0 = root(ns[0])
        for g in ns[1:]:
            par[root(g)] = r0
    nel.append(ne)
    ncomp.append(len({root(g) for g in par}))
    sh = f.replace(".nodes", ".shared")
    nshared.append(sum(1 for _ in open(sh)) if os.path.exists(sh) else 0)
ys = np.array([v[1] for v in xy.values()]); ymin, ymax = ys.min(), ys.max(); eps = 1e-5 * (ymax - ymin)
bot = {round(x, 3): g for g, (x, y) in xy.items() if abs(y - ymin) < eps}
top = {round(x, 3): g for g, (x, y) in xy.items() if abs(y - ymax) < eps}
pairs = [(top[x], bot[x]) for x in top if x in bot]
unpaired = len(top) - len(pairs)
# by the elements: the partitions whose elements use the node (the nodes file may also list a counterpart that no
# element of that partition uses: a ghost)
U = lambda g: used.get(g, set())
bad = [(s, m) for s, m in pairs if not U(s) <= U(m)]
split = sum(1 for s, m in pairs if U(s) != U(m))
ghosts = sum(len(where[g] - U(g)) for g in where)
nel = np.array(nel)
print(f"{len(files)} partitions: elements per partition {nel.min()}..{nel.max()} (imbalance {nel.max() / nel.mean():.3f}), "
      f"shared nodes per partition {min(nshared)}..{max(nshared)} (total {sum(nshared)})")
print(f"periodic pairs {len(pairs)} (unpaired top nodes {unpaired}): slave in a partition without its master: {len(bad)}; "
      f"pairs with different partition sets: {split}; ghost node copies (listed, no element): {ghosts}")
# a partition touching a periodic side at a single node, with no side edge there (none of its elements has a second
# node on that side next to it): Elmer builds the periodic constraint from the boundary elements of each partition, so
# that partition has none for the pair (720 ranks, 50 km mesh: the one non-periodic column)
side = lambda g: abs(xy[g][1] - ymin) < eps or abs(xy[g][1] - ymax) < eps
edgep = {}                                   # side node -> partitions with an element having a side edge there
vtouch = {}                                  # side node -> partitions using it only through single-node elements
for f in files:
    p = int(re.findall(r"part\.(\d+)\.nodes", f)[0])
    for line in open(f.replace(".nodes", ".elements")):
        ns = [int(q) for q in line.split()[3:]]
        sn = [g for g in ns if g in xy and side(g)]
        if len(sn) >= 2:
            for g in sn:
                edgep.setdefault(g, set()).add(p)
        elif len(sn) == 1:
            vtouch.setdefault(sn[0], set()).add(p)
vonly = [(g, sorted(ps - edgep.get(g, set()))) for g, ps in vtouch.items() if ps - edgep.get(g, set())]
print(f"side nodes used by a partition without a side edge there (vertex-only touch): {len(vonly)}"
      + ("".join(f"\n   node {g} at x {xy[g][0] / 1e3:.2f} km, y {'0' if abs(xy[g][1] - ymin) < eps else 'W'}: partitions {ps}"
                 for g, ps in sorted(vonly)[:10])))
bad = bad if not vonly else bad + [(g, g) for g, _ in vonly]
ncomp = np.array(ncomp)
print(f"connected pieces per partition: max {ncomp.max()}; partitions in more than one piece: {(ncomp > 1).sum()}"
      + (f" ({', '.join(str(i + 1) + ':' + str(c) for i, c in enumerate(ncomp) if c > 1)[:200]})" if (ncomp > 1).any() else ""))
sh = np.array(nshared); print(f"shared nodes / sqrt(elements) (compactness, larger = more ragged): median "
      f"{np.median(sh / np.sqrt(nel)):.2f}, max {(sh / np.sqrt(nel)).max():.2f} (partition {np.argmax(sh / np.sqrt(nel)) + 1})")
for s, m in bad[:5]:
    print(f"   slave {s} at x {xy[s][0]:.1f} used in {sorted(U(s))}, master {m} used in {sorted(U(m))}")
raise SystemExit(1 if bad else 0)
