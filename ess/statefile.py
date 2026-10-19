"""
Binary state files shared with the Elmer solver SteadyStateIO.F90, and the state layout.

File format (stream, little-endian):
    int32 magic, int32 version, int32 nrec
    nrec x { int32 namelen, char[namelen] name, int64 n, int64[n] gid, float64[n] value }

Elmer writes one file per partition (<base>.<pe>); the driver merges them by global node id.
Elmer reads one merged file whose gids are sorted.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

MAGIC = 1162089265
VERSION = 1

Records = dict  # name -> (gids int64 sorted, values float64)


def write_state_file(path: str, records: Records) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.array([MAGIC, VERSION, len(records)], "<i4").tofile(f)
        for name, (g, v) in records.items():
            g = np.ascontiguousarray(g, "<i8")
            v = np.ascontiguousarray(v, "<f8")
            if g.shape != v.shape or g.ndim != 1:
                raise ValueError(f"record {name}: gid/value shape mismatch")
            if len(g) > 1 and np.any(np.diff(g) <= 0):
                raise ValueError(f"record {name}: gids must be strictly increasing")
            b = name.encode()
            np.array([len(b)], "<i4").tofile(f)
            f.write(b)
            np.array([len(g)], "<i8").tofile(f)
            g.tofile(f)
            v.tofile(f)
    os.replace(tmp, path)


def read_state_file(path: str) -> Records:
    out = {}
    with open(path, "rb") as f:
        hdr = np.fromfile(f, "<i4", 3)
        if len(hdr) < 3 or hdr[0] != MAGIC:
            raise ValueError(f"{path}: not an ESS state file")
        if hdr[1] != VERSION:
            raise ValueError(f"{path}: unsupported version {hdr[1]}")
        for _ in range(int(hdr[2])):
            nlen = int(np.fromfile(f, "<i4", 1)[0])
            name = f.read(nlen).decode()
            n = int(np.fromfile(f, "<i8", 1)[0])
            g = np.fromfile(f, "<i8", n)
            v = np.fromfile(f, "<f8", n)
            if len(g) != n or len(v) != n:
                raise ValueError(f"{path}: truncated record {name}")
            out[name] = (g, v)
    return out


def partition_files(base: str) -> list[str]:
    files = [p for p in glob.glob(glob.escape(base) + ".*") if re.fullmatch(r"\d+", p.rsplit(".", 1)[1])]
    return sorted(files, key=lambda p: int(p.rsplit(".", 1)[1]))


_NONFINITE_SEEN: set = set()


def merge_partitions(paths: list[str], rtol: float = 1e-9) -> tuple[Records, dict]:
    """Merge per-partition files by global id. Returns records and the max mismatch between duplicate
    (shared-node) values of each variable, relative to the variable's max magnitude."""
    parts = [read_state_file(p) for p in paths]
    names = []
    for p in parts:
        for k in p:
            if k not in names:
                names.append(k)
    out, mism = {}, {}
    for k in names:
        g = np.concatenate([p[k][0] for p in parts if k in p])
        v = np.concatenate([p[k][1] for p in parts if k in p])
        order = np.lexsort((v, g))
        g, v = g[order], v[order]
        first = np.ones(len(g), bool)
        first[1:] = g[1:] != g[:-1]
        idx = np.cumsum(first) - 1
        gu, vu = g[first], v[first]
        with np.errstate(invalid="ignore"):          # diagnostics may hold NaN (e.g. never-set variables)
            d = np.abs(v - vu[idx])
        fin = np.isfinite(vu)
        if not fin.all() and k not in _NONFINITE_SEEN:
            _NONFINITE_SEEN.add(k)
            import logging
            logging.getLogger("ess").info(f"variable '{k}': {int((~fin).sum())} non-finite values in Elmer's "
                                          f"output (reported once)")
        scale = max(float(np.max(np.abs(vu[fin]))) if fin.any() else 0.0, 1e-300)
        mism[k] = float(np.nanmax(d) / scale) if len(d) and np.isfinite(d).any() else 0.0
        out[k] = (gu, vu)
    return out, mism


@dataclass
class Block:
    name: str
    gids: np.ndarray
    scale: float = 1.0


@dataclass
class StateLayout:
    """The ordered list of nodal blocks making up the state vector x (physical units)."""
    blocks: list[Block] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(len(b.gids) for b in self.blocks)

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.blocks]

    def slices(self) -> dict[str, slice]:
        out, i = {}, 0
        for b in self.blocks:
            out[b.name] = slice(i, i + len(b.gids))
            i += len(b.gids)
        return out

    def scale_vector(self) -> np.ndarray:
        return np.concatenate([np.full(len(b.gids), float(b.scale)) for b in self.blocks])

    @classmethod
    def from_records(cls, records: Records, names: list[str], scales: dict[str, float]) -> "StateLayout":
        blocks = []
        for n in names:
            key = find_name(records, n)
            blocks.append(Block(n, np.asarray(records[key][0], np.int64).copy(), float(scales.get(n, 1.0))))
        return cls(blocks)

    def to_vector(self, records: Records) -> np.ndarray:
        parts = []
        for b in self.blocks:
            key = find_name(records, b.name)
            g, v = records[key]
            if len(g) != len(b.gids) or not np.array_equal(g, b.gids):
                missing = np.setdiff1d(b.gids, g)
                extra = np.setdiff1d(g, b.gids)
                raise ValueError(f"block {b.name}: node set differs from the layout "
                                 f"({len(missing)} missing, {len(extra)} extra)")
            parts.append(np.asarray(v, float))
        return np.concatenate(parts)

    def to_records(self, x: np.ndarray) -> Records:
        if len(x) != self.size:
            raise ValueError(f"state has {len(x)} entries, layout {self.size}")
        sl = self.slices()
        return {b.name: (b.gids, np.asarray(x[sl[b.name]], float)) for b in self.blocks}

    def split(self, x: np.ndarray) -> dict[str, np.ndarray]:
        sl = self.slices()
        return {k: x[s] for k, s in sl.items()}

    def to_npz_dict(self, prefix="layout_") -> dict:
        d = {prefix + "names": np.array(self.names), prefix + "scales": np.array([b.scale for b in self.blocks])}
        for i, b in enumerate(self.blocks):
            d[f"{prefix}gids_{i}"] = b.gids
        return d

    @classmethod
    def from_npz(cls, z, prefix="layout_") -> "StateLayout":
        names = [str(n) for n in z[prefix + "names"]]
        scales = z[prefix + "scales"]
        return cls([Block(n, z[f"{prefix}gids_{i}"], float(scales[i])) for i, n in enumerate(names)])


def find_name(records: Records, name: str) -> str:
    """Elmer may change the case of variable names; match case-insensitively."""
    if name in records:
        return name
    low = {k.lower(): k for k in records}
    if name.lower() in low:
        return low[name.lower()]
    raise KeyError(f"variable '{name}' not in state file (have: {', '.join(records)})")


# ---- driver-side state files (.npz): state vector + layout + frozen fields + parameter

def save_state(path: str, layout: StateLayout, x: np.ndarray, p: float, frozen: Records | None = None,
               diag: Records | None = None, **meta) -> None:
    d = layout.to_npz_dict()
    d["x"] = np.asarray(x, float)
    d["p"] = np.float64(p)
    for k, (g, v) in (frozen or {}).items():
        d[f"frozen_gids::{k}"] = g
        d[f"frozen_vals::{k}"] = v
    for k, (g, v) in (diag or {}).items():
        d[f"diag_gids::{k}"] = g
        d[f"diag_vals::{k}"] = v
    for k, v in meta.items():
        d[f"meta::{k}"] = np.asarray(v)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)


@dataclass
class SavedState:
    layout: StateLayout
    x: np.ndarray
    p: float
    frozen: Records
    diag: Records
    meta: dict


def load_state(path: str) -> SavedState:
    with np.load(path, allow_pickle=False) as z:
        layout = StateLayout.from_npz(z)
        frozen, diag, meta = {}, {}, {}
        for k in z.files:
            if k.startswith("frozen_gids::"):
                n = k.split("::", 1)[1]
                frozen[n] = (z[k], z["frozen_vals::" + n])
            elif k.startswith("diag_gids::"):
                n = k.split("::", 1)[1]
                diag[n] = (z[k], z["diag_vals::" + n])
            elif k.startswith("meta::"):
                meta[k.split("::", 1)[1]] = z[k][()] if z[k].ndim == 0 else z[k]
        return SavedState(layout, z["x"].copy(), float(z["p"]), frozen, diag, meta)
