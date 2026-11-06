#!/usr/bin/env python3
"""Compare the three rotational periodic test runs (run_rotper.sh): periodic, walls, cartesian.

    python3 periodic_test/rotper/compare_rotper.py WORKDIR ANGLE

The exact solution is axisymmetric: no azimuthal velocity, and the same radial and vertical velocity at every angle.
"walls" (free-slip sides) is that solution on this mesh, so:
  - periodic should match walls to discretisation round-off, with v_theta ~ 0 everywhere and the two sides equal in
    (v_r, v_theta, v_z);
  - cartesian (vx, vy, vz tied across the seam) should not: it forces the wrong velocity direction on one side.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plots"))
from plot_state_3d import read_vtu  # noqa: E402

wd, angle = sys.argv[1], float(sys.argv[2])
half = np.radians(angle) / 2


def load(case):
    fs = sorted(glob.glob(os.path.join(wd, f"rotper_{case}*.pvtu")) + glob.glob(os.path.join(wd, "sector", f"rotper_{case}*.pvtu")))
    if not fs:
        fs = sorted(glob.glob(os.path.join(wd, f"rotper_{case}*.vtu")) + glob.glob(os.path.join(wd, "sector", f"rotper_{case}*.vtu")))
        fs = [f for f in fs if "_par" not in os.path.basename(f)]
    if not fs:
        return None
    X, D = read_vtu(fs[-1])
    u = D["velocity"]
    key = np.round(X, 1)
    _, first = np.unique(key, axis=0, return_index=True)          # shared nodes once
    X, u = X[first], u[first]
    r = np.hypot(X[:, 0], X[:, 1])
    ph = np.arctan2(X[:, 1], X[:, 0])
    vr = u[:, 0] * np.cos(ph) + u[:, 1] * np.sin(ph)
    vt = -u[:, 0] * np.sin(ph) + u[:, 1] * np.cos(ph)
    return dict(X=X, r=r, ph=ph, z=X[:, 2], vr=vr, vt=vt, vz=u[:, 2], file=fs[-1])


R = {c: load(c) for c in ("walls", "periodic", "cartesian")}
if R["walls"] is None:
    raise SystemExit(f"{wd}: no rotper_walls output")
W = R["walls"]
umax = np.max(np.abs(W["vr"]))
print(f"walls (the axisymmetric solution): max |v_r| {umax:.3e}, max |v_theta| {np.max(np.abs(W['vt'])):.3e}")
ref = {tuple(k): i for i, k in enumerate(np.round(W["X"], 1))}

for case in ("periodic", "cartesian"):
    P = R[case]
    if P is None:
        print(f"{case}: no output")
        continue
    idx = np.array([ref.get(tuple(k), -1) for k in np.round(P["X"], 1)])
    ok = idx >= 0
    d = np.sqrt((P["vr"][ok] - W["vr"][idx[ok]]) ** 2 + (P["vt"][ok] - W["vt"][idx[ok]]) ** 2
                + (P["vz"][ok] - W["vz"][idx[ok]]) ** 2)
    # the seam: side 1 (phi = -angle/2) against side 2 (+angle/2) at the same r, z
    s1 = np.abs(P["ph"] + half) < 1e-6
    s2 = np.abs(P["ph"] - half) < 1e-6
    k1 = {(round(r, 1), round(z, 1)): i for i, (r, z) in enumerate(zip(P["r"], P["z"])) if s1[i]}
    pairs = [(k1[(round(r, 1), round(z, 1))], j) for j, (r, z) in enumerate(zip(P["r"], P["z"]))
             if s2[j] and (round(r, 1), round(z, 1)) in k1]
    seam = max((max(abs(P[c][a] - P[c][b]) for c in ("vr", "vt", "vz")) for a, b in pairs), default=np.nan)
    print(f"{case}: {ok.sum()} nodes matched; |u - u_walls| max {d.max() / umax:.2e}, rms {np.sqrt(np.mean(d ** 2)) / umax:.2e}"
          f" (of max |v_r|); max |v_theta| {np.max(np.abs(P['vt'])) / umax:.2e}; seam pairs {len(pairs)}, "
          f"max side-1/side-2 difference in (v_r, v_theta, v_z) {seam / umax:.2e}")
print("expected: periodic ~1e-10 or below everywhere (the walls solution exactly); cartesian visibly off "
      "(v_theta of order sin(angle/2) of v_r near the seam)")
