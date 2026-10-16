#!/usr/bin/env python3
"""
Stand-in for `mpirun -np N ElmerSolver run.sif`, to test the driver's Elmer plumbing without Elmer.

    fake_elmer.py N ElmerSolver run.sif   (as launched by mpirun = "python fake_elmer.py {n}")

Reads the generated sif the way the real template is laid out (StateRead file, timestep settings,
StateWrite variables, restart block), reads parameters.lua for Tsurf, and runs a simple nodal model on
a fake partitioned "mesh": node gids 1..M, Temperature on all nodes, Zs on the top nodes, masks/Zb/
bedrock on the bottom nodes. Each of the N "partitions" writes its own output file, with overlapping
(shared) nodes, as Elmer does.

Dynamics (backward Euler, exact fixed point known):
    dT/dt = -k (T - Tstar),  Tstar = Tsurf + 2 + 0.1 sin(gid) + 0.5 tanh(Zs_mean - 100)
    dZs/dt = -kz (Zs - 100 - 0.01 gid)
"""
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from ess.statefile import read_state_file, write_state_file  # noqa: E402

NX, NZ = 12, 6
M = NX * NZ
GID = np.arange(1, M + 1)
TOP = GID[GID % NZ == 0]            # every NZ-th node is a surface node
BOT = GID[GID % NZ == 1]


def solver_block(sif, eq):
    m = re.search(r"Solver\s+\d+\s*\n(?:(?!\nEnd).)*?Equation = \"" + re.escape(eq) + r"\"(?:(?!\nEnd).)*", sif, re.S)
    return m.group(0) if m else ""


def val(block, key):
    m = re.search(re.escape(key) + r'\s*=\s*(?:File|String)?\s*"([^"]*)"', block)
    return m.group(1) if m else None


def tstar(Tsurf, zs):
    return Tsurf + 2 + 0.1 * np.sin(GID) + 0.5 * np.tanh(np.mean(zs) - 100)


def main():
    n = int(sys.argv[1])
    sif = open(sys.argv[-1]).read()
    params = open("parameters.lua").read()
    Tsurf = float(re.search(r"^\s*Tsurf\s*=\s*([-+0-9.eE]+)", params, re.M).group(1))
    nsteps = int(re.search(r"Timestep Intervals\s*=\s*(\d+)", sif).group(1))
    dt = float(re.search(r"Timestep Sizes\s*=\s*([-+0-9.eE]+)", sif).group(1))
    fin = val(solver_block(sif, "StateRead"), "State File")
    fout0 = val(solver_block(sif, "StateWriteInitial"), "State File")
    wblock = solver_block(sif, "StateWrite")
    fout = val(wblock, "State File")
    names = re.findall(r'State Variable \d+ = String "([^"]*)"', wblock)
    names0 = re.findall(r'State Variable \d+ = String "([^"]*)"', solver_block(sif, "StateWriteInitial"))

    # initial fields (the "restart" / initial conditions)
    f = {"temperature": 260 + 0.01 * GID.astype(float), "zs": 90 + 0 * TOP.astype(float),
         "coldtempmask": np.where(BOT < M / 2, -1.0, 1.0), "groundedmask": np.ones(len(BOT)),
         "zb": -0.1 * BOT.astype(float), "bedrock": -0.1 * BOT.astype(float)}
    gids = {"temperature": GID, "zs": TOP, "coldtempmask": BOT, "groundedmask": BOT, "zb": BOT, "bedrock": BOT}
    if fin and fin != "none":
        rec = read_state_file(fin)
        for k, (g, v) in rec.items():
            kk = k.lower()
            if not np.array_equal(g, gids[kk]):
                print(f"ERROR StateIO: node set mismatch for {k}")
                sys.exit(1)
            f[kk] = v.copy()

    def write(base, vlist):
        for r in range(n):
            own = (GID % n == r) | (GID % n == (r + 1) % n) & (GID < 10)     # some shared nodes
            rec = {}
            for v in vlist:
                kk = v.lower()
                if kk == "temperature loads":
                    g, val_ = BOT, -1.0 + 0 * BOT
                elif kk in f:
                    g, val_ = gids[kk], f[kk]
                else:
                    continue
                sel = own[g - 1]
                rec[v] = (g[sel], val_[sel])
            for c in (1, 2, 3):
                rec[f"coordinate {c}"] = (GID[own], (GID[own] * (c == 1) + (GID[own] % NZ) * (c == 3)).astype(float))
            write_state_file(f"{base}.{r}", rec)

    if fout0 and fout0 != "none":
        write(fout0, names0)
    T, zs = f["temperature"], f["zs"]
    for _ in range(nsteps):
        T = (T + dt * 1.0 * tstar(Tsurf, zs)) / (1 + dt * 1.0)       # staggered: Tstar from old Zs
        zs = (zs + dt * 0.5 * (100 + 0.01 * TOP)) / (1 + dt * 0.5)
    f["temperature"], f["zs"] = T, zs
    write(fout, names)
    print("*** Elmer Solver: ALL DONE ***")


if __name__ == "__main__":
    main()
