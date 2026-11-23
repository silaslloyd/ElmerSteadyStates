#!/usr/bin/env python3
"""An Elmer VTU output step (e.g. the last step of an ElmerStreams flowline run) as an ESS SOURCE state for
`ess transfer`: its temperature (every node: ice, filler, bedrock), Zs (the top nodes), Zb (the base nodes), the
grounded and cold/temperate masks (the base nodes' GroundedMask / ColdTempMask fields; without them: grounded where
Zb is within 1 m of the bedrock field, temperate where the base temperature is within 0.01 K of 273.15), the node
coordinates. ess transfer then carries it onto any of our meshes column by column (ice on the normalised height,
rock on the depth below the base) and takes the masks from it.

Elmer writes 0 where a variable is not defined: Zs on the nonzero (top) nodes, Zb / the masks on the base nodes
(Zb nonzero; Max Zb is below sea level, so a real Zb is never 0).

    python3 jobs/vtu_to_state.py VTU_DIR_OR_FILE OUT.npz [--p 268.15]
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "plots"))
from vtu_profile import field, pick, pieces_of, read_vtu  # noqa: E402
from ess.statefile import Block, StateLayout, save_state  # noqa: E402


def main():
    a = sys.argv[1:]
    p = 268.15
    if "--p" in a:
        i = a.index("--p")
        p = float(a[i + 1])
        del a[i:i + 2]
    if len(a) != 2:
        raise SystemExit(__doc__)
    src, out = a
    files, what = (pieces_of(src), src) if os.path.isfile(src) else pick(src)
    print(f"reading {what}")
    P, F = [], {}
    for f in files:
        r = read_vtu(f)
        P.append(r["points"])
        for k, v in r["fields"].items():
            F.setdefault(k, []).append(v)
    P = np.concatenate(P)
    F = {k: np.concatenate(v) for k, v in F.items() if sum(len(x) for x in v) == len(P)}
    print(f"{len(P)} nodes; fields {', '.join(sorted(F))}")
    # partition interface nodes appear once per piece: keep one copy per position
    key = np.round(P, 3)
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    P = P[first]
    F = {k: (v[first]) for k, v in F.items()}
    n = len(P)
    g = np.arange(1, n + 1, dtype=np.int64)
    kT, ks, kb = field(F, "temperature"), field(F, "zs"), field(F, "zb")
    kbed, kgm, kct = field(F, "bedrock"), field(F, "groundedmask"), field(F, "coldtempmask")
    if not (kT and ks and kb):
        raise SystemExit(f"need Temperature, Zs and Zb point fields; have {sorted(F)}")
    top = F[ks] != 0
    base = F[kb] != 0
    if kgm:
        gm = F[kgm][base]
        how_g = f"field {kgm!r}"
    elif kbed:
        gm = np.where(np.abs(F[kb][base] - F[kbed][base]) < 1.0, 1.0, -1.0)
        how_g = "Zb within 1 m of the bedrock field"
    else:
        raise SystemExit("no GroundedMask or bedrock field")
    if kct:
        ct = np.where(F[kct][base] > 0, 1.0, -1.0)
        how_c = f"field {kct!r}"
    else:
        ct = np.where(F[kT][base] > 273.15 - 0.01, 1.0, -1.0)
        how_c = "base temperature within 0.01 K of melting"
    ct = np.where(gm < -0.5, 1.0, ct)                 # floating base: temperate (at T0), as our states
    layout = StateLayout([Block("Temperature", g.copy(), 5.0), Block("Zs", g[top].copy(), 50.0),
                          Block("Zb", g[base].copy(), 50.0)])
    x = np.concatenate([F[kT], F[ks][top], F[kb][base]])
    frozen = {"ColdTempMask": (g[base], ct), "GroundedMask": (g[base], gm)}
    if kbed:
        frozen["bedrock"] = (g[base], F[kbed][base])
    khw = field(F, "water sheet thickness")
    if khw:                                       # the run's own water sheet (carried by ess transfer as the masks)
        frozen["Water Sheet Thickness"] = (g[base], F[khw][base])
        print(f"water sheet from {khw!r}: {F[khw][base].min():.3g} .. {F[khw][base].max():.3g} m on the base")
    diag = {f"coordinate {c + 1}": (g, P[:, c]) for c in range(3)}
    xb = P[base, 0]
    xgl = xb[gm > -0.5].max() / 1e3 if (gm > -0.5).any() else np.nan
    save_state(out, layout, x, p, frozen, diag, source=what)
    print(f"{n} nodes ({top.sum()} top, {base.sum()} base); grounded mask from {how_g}, cold/temperate from {how_c}; "
          f"grounding line {xgl:.2f} km, temperate grounded base nodes {int(((gm > -0.5) & (ct > 0)).sum())} of "
          f"{int((gm > -0.5).sum())} -> {out}")


if __name__ == "__main__":
    main()
