#!/usr/bin/env python3
"""The upper and lower ice surface along x from the LAST time step of an Elmer VTU output directory (e.g. an
ElmerStreams run), saved as a small profile (npz: x_km, zs, zb, ...) for plots/plot_section.py --overlay.

Which files: every .pvtu under DIR (recursively), the one with the highest time-step number in its name (the last
integer; ties by modification time); without .pvtu files, the .vtu files grouped by their step number (t0123 or the
last integer) and the highest step's pieces read together. --file F.pvtu|F.vtu: that file instead. The choice is printed.

What (an extruded mesh: nodes in vertical columns): the upper surface is the top node of each column (the mesh
follows it); the lower surface the Zb point field there (any case: "zb", also "fs lower" / "lower surface"; its nonzero
values: Elmer writes 0 where a variable is not defined), else (with a warning) the column's bottom node, which is the
bed, not the ice base, if the output holds the bedrock too. 3D output: averaged across y at each x; --y Y0 Y1 (km):
only that band.

A pure-Python VTU reader (no vtk / meshio needed): ascii, binary (base64, uncompressed) and appended raw data (Elmer's
default), zlib-compressed or not.

    python3 plots/vtu_profile.py DIR [--file F] [--out PROFILE.npz] [--y Y0 Y1] [--name LABEL]
"""
import base64
import glob
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np

NP = {"Float32": "<f4", "Float64": "<f8", "Int8": "<i1", "UInt8": "<u1", "Int16": "<i2", "UInt16": "<u2",
      "Int32": "<i4", "UInt32": "<u4", "Int64": "<i8", "UInt64": "<u8"}


def read_vtu(path):
    """{'points': (n, 3), 'fields': {name: array}} of one .vtu piece"""
    data = open(path, "rb").read()
    raw = b""
    i = data.find(b"<AppendedData")
    if i >= 0:
        j = data.index(b"_", data.index(b">", i)) + 1
        raw = data[j:]
        xml = data[:i] + b"</VTKFile>"
    else:
        xml = data
    root = ET.fromstring(xml)
    hdr = NP[root.get("header_type", "UInt32")]
    hsz = np.dtype(hdr).itemsize
    comp = root.get("compressor")
    if root.get("byte_order", "LittleEndian") != "LittleEndian":
        raise SystemExit(f"{path}: big-endian VTU not supported")

    def decode(da):
        dt = NP[da.get("type")]
        fmt = da.get("format", "ascii")
        if fmt == "ascii":
            return np.array((da.text or "").split(), dtype=dt)
        if fmt == "appended":
            off = int(da.get("offset"))
            if comp:
                nb, bs, last = np.frombuffer(raw, hdr, 3, off)
                csz = np.frombuffer(raw, hdr, int(nb), off + 3 * hsz)
                p = off + (3 + int(nb)) * hsz
                out = []
                for c in csz:
                    out.append(zlib.decompress(raw[p:p + int(c)]))
                    p += int(c)
                return np.frombuffer(b"".join(out), dt)
            n = int(np.frombuffer(raw, hdr, 1, off)[0])
            return np.frombuffer(raw, dt, n // np.dtype(dt).itemsize, off + hsz)
        if fmt == "binary":
            if comp:
                raise SystemExit(f"{path}: compressed inline binary VTU not supported (appended or ascii are)")
            b = base64.b64decode("".join((da.text or "").split()))
            n = int(np.frombuffer(b, hdr, 1, 0)[0])
            return np.frombuffer(b, dt, n // np.dtype(dt).itemsize, hsz)
        raise SystemExit(f"{path}: DataArray format {fmt!r}")

    piece = root.find("UnstructuredGrid/Piece")
    pts = decode(piece.find("Points/DataArray")).astype(float).reshape(-1, 3)
    fields = {}
    pd = piece.find("PointData")
    if pd is not None:
        for da in pd.findall("DataArray"):
            v = decode(da).astype(float)
            nc = int(da.get("NumberOfComponents", "1"))
            fields[da.get("Name")] = v.reshape(-1, nc) if nc > 1 else v
    return {"points": pts, "fields": fields}


def step_of(path):
    m = re.search(r"t(\d+)\D*$", os.path.basename(path)) or re.search(r"(\d+)\D*$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def pick(dirn):
    """the last time step's pieces: (list of .vtu paths, description)"""
    pv = sorted(set(glob.glob(os.path.join(dirn, "**", "*.pvtu"), recursive=True)))
    if pv:
        f = max(pv, key=lambda p: (step_of(p), os.path.getmtime(p)))
        return pieces_of(f), f"{f} (step {step_of(f)}, of {len(pv)} .pvtu)"
    vt = sorted(set(glob.glob(os.path.join(dirn, "**", "*.vtu"), recursive=True)))
    if not vt:
        raise SystemExit(f"no .pvtu or .vtu files under {dirn}")
    s = max(step_of(p) for p in vt)
    sel = sorted(p for p in vt if step_of(p) == s)
    return sel, f"{len(sel)} .vtu piece(s) of step {s} (e.g. {sel[0]})"


def pieces_of(f):
    if f.endswith(".pvtu"):
        root = ET.parse(f).getroot()
        return [os.path.join(os.path.dirname(f), p.get("Source")) for p in root.iter("Piece")]
    return [f]


def field(fields, *names):
    low = {k.lower(): k for k in fields}
    for n in names:
        if n in low:
            return low[n]
    return None


def main():
    a = sys.argv[1:]

    def opt(name, n=1, conv=str):
        if name not in a:
            return None
        i = a.index(name)
        v = a[i + 1:i + 1 + n]
        del a[i:i + 1 + n]
        return [conv(x) for x in v] if n > 1 else conv(v[0])
    f = opt("--file")
    out = opt("--out")
    yb = opt("--y", 2, float)
    name = opt("--name")
    if not a and not f:
        raise SystemExit(__doc__)
    dirn = a[0] if a else os.path.dirname(f)
    files, what = (pieces_of(f), f) if f else pick(dirn)
    print(f"reading {what}")
    P, F = [], {}
    for p in files:
        r = read_vtu(p)
        P.append(r["points"])
        for k, v in r["fields"].items():
            F.setdefault(k, []).append(v)
    P = np.concatenate(P)
    F = {k: np.concatenate(v) for k, v in F.items() if sum(len(x) for x in v) == len(P)}
    print(f"{len(P)} nodes; point fields: {', '.join(sorted(F))}")
    x, y, z = P[:, 0] / 1e3, P[:, 1] / 1e3, P[:, 2]
    keep = np.ones(len(x), bool)
    if yb:
        keep &= (y >= yb[0]) & (y <= yb[1])
    kb = field(F, "zb", "fs lower", "lower surface", "lower_surface")
    xk, yk, zk = x[keep], y[keep], z[keep]
    zbv = F[kb][keep] if kb else None
    # columns: nodes sharing (x, y) to 1 m (an extruded mesh); the upper surface is the top node of each column
    # (the mesh follows it), the lower surface the Zb field there (Elmer writes 0 where a variable is not defined:
    # the nonzero values), else the column's bottom node
    col = np.round(xk * 1e3).astype(np.int64) * 10 ** 8 + np.round(yk * 1e3).astype(np.int64)
    uc, ci = np.unique(col, return_inverse=True)
    if len(uc) > len(col) / 3:
        raise SystemExit("the nodes do not form vertical columns (an extruded mesh): not handled")
    top = np.full(len(uc), -np.inf)
    np.maximum.at(top, ci, zk)
    bot = np.full(len(uc), np.inf)
    np.minimum.at(bot, ci, zk)
    if zbv is not None:
        nz_ = zbv != 0
        cnt = np.bincount(ci[nz_], minlength=len(uc))
        sm = np.bincount(ci[nz_], weights=zbv[nz_], minlength=len(uc))
        base = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
        src = f"Zs: column tops; Zb: field {kb!r} (its nonzero nodes)"
    else:
        print("WARNING: no Zb point field: the bottom node of each column instead (the bed, if the output holds the "
              "bedrock)")
        base = bot
        src = "Zs: column tops; Zb: column bottoms (no Zb field)"
    cx = (uc // 10 ** 8) / 1e3                          # the column's x (km)
    xc, xi = np.unique(np.round(cx, 3), return_inverse=True)
    n_ = np.bincount(xi)
    zs = np.bincount(xi, weights=top) / n_             # mean across y
    okb = np.isfinite(base)
    zb = np.bincount(xi[okb], weights=base[okb], minlength=len(xc)) / np.maximum(np.bincount(xi[okb], minlength=len(xc)), 1)
    zb[np.bincount(xi[okb], minlength=len(xc)) == 0] = np.nan
    print(f"{len(uc)} columns at {len(xc)} x positions ({int(n_.max())} across y at most)")
    ok = np.isfinite(zs) & np.isfinite(zb)
    out = out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                              re.sub(r"\W+", "_", os.path.basename(os.path.normpath(dirn))) + "_profile.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, x_km=xc[ok], zs=zs[ok], zb=zb[ok], source=what, how=src,
             name=name or os.path.basename(os.path.normpath(dirn)))
    print(f"Zs, Zb from {src}: {ok.sum()} x positions, {xc[ok].min():.1f} .. {xc[ok].max():.1f} km; "
          f"Zs max {np.nanmax(zs):.0f} m -> {out}")


if __name__ == "__main__":
    main()
