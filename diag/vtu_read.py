#!/usr/bin/env python3
"""A dependency-free (numpy only) reader for Elmer's .vtu files: ascii, inline base64 binary (optionally zlib), and
appended data (raw or base64). read_vtu(path) -> dict(points (n,3), cells (connectivity, offsets, types), point (name ->
array), cell (name -> array)).  read_many(paths) concatenates partitions (points and point fields only)."""
import base64
import re
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np

DT = {"Float32": "f4", "Float64": "f8", "Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2", "Int32": "i4",
      "UInt32": "u4", "Int64": "i8", "UInt64": "u8"}


def _decode_blocks(buf, hdr, compressed):
    """buf: raw bytes starting at the header (VTK block header + payload) -> payload bytes"""
    h = np.dtype("u%d" % hdr)
    if not compressed:
        n = int(np.frombuffer(buf[:hdr], h)[0])
        return buf[hdr:hdr + n]
    nb, bs, last = (int(v) for v in np.frombuffer(buf[:3 * hdr], h))
    sizes = np.frombuffer(buf[3 * hdr:(3 + nb) * hdr], h).astype(int)
    pos, out = (3 + nb) * hdr, []
    for s in sizes:
        out.append(zlib.decompress(buf[pos:pos + s]))
        pos += s
    return b"".join(out)


def _b64_inline(text, hdr, compressed):
    """inline base64: the VTK standard encodes the block header and the payload separately; some writers encode them as
    one string. Try the separate form first and fall back to the joined one when it is not self-consistent."""
    text = "".join(text.split())
    for joined in (False, True):
        try:
            if joined:
                buf = base64.b64decode(text)
                out = _decode_blocks(buf, hdr, compressed)
                return out
            if not compressed:
                k = -(-hdr // 3) * 4
                n = int(np.frombuffer(base64.b64decode(text[:k]), "u%d" % hdr)[0])
                out = base64.b64decode(text[k:])
                if len(out) < n:
                    raise ValueError("short")
                return out[:n]
            k = -(-3 * hdr // 3) * 4
            nb = int(np.frombuffer(base64.b64decode(text[:k]), "u%d" % hdr, count=3)[0])
            kh = -(-(3 + nb) * hdr // 3) * 4
            head = base64.b64decode(text[:kh])
            sizes = np.frombuffer(head[3 * hdr:(3 + nb) * hdr], "u%d" % hdr).astype(int)
            data = base64.b64decode(text[kh:])
            pos, out = 0, []
            for sz in sizes:
                out.append(zlib.decompress(data[pos:pos + sz]))
                pos += sz
            return b"".join(out)
        except Exception:
            if joined:
                raise
    raise ValueError("cannot decode the inline data")


def read_vtu(path):
    raw = open(path, "rb").read()
    app = None
    m = re.search(rb"<AppendedData[^>]*encoding=\"(\w+)\"[^>]*>\s*_", raw)
    if m:
        app = (m.group(1).decode(), raw[m.end():raw.rindex(b"</AppendedData>")])
        xml = raw[:m.start()] + b"</AppendedData></VTKFile>"
    else:
        xml = raw
    root = ET.fromstring(xml)
    hdr = 8 if root.get("header_type") == "UInt64" else 4
    compressed = root.get("compressor") is not None

    def arr(da):
        t = DT[da.get("type")]
        nc = int(da.get("NumberOfComponents", "1"))
        fmt = da.get("format", "ascii")
        if fmt == "ascii":
            a = np.array((da.text or "").split(), dtype=t)
        elif fmt == "binary":
            a = np.frombuffer(_b64_inline(da.text, hdr, compressed), dtype=t)
        else:                                    # appended
            off = int(da.get("offset"))
            enc, data = app
            if enc == "raw":
                a = np.frombuffer(_decode_blocks(data[off:], hdr, compressed), dtype=t)
            else:
                a = np.frombuffer(_b64_inline(data[off:].decode(), hdr, compressed), dtype=t)
        return a.reshape(-1, nc) if nc > 1 else a

    piece = root.find(".//Piece")
    pts = arr(piece.find("Points/DataArray"))
    out = {"points": pts, "point": {}, "cell": {}}
    for da in piece.findall("PointData/DataArray"):
        out["point"][da.get("Name")] = arr(da)
    for da in piece.findall("CellData/DataArray"):
        out["cell"][da.get("Name")] = arr(da)
    c = {da.get("Name"): arr(da) for da in piece.findall("Cells/DataArray")}
    out["cells"] = c
    return out


def read_many(paths):
    P, pt = [], {}
    for f in paths:
        d = read_vtu(f)
        P.append(d["points"])
        for k, v in d["point"].items():
            pt.setdefault(k, []).append(v)
    return np.vstack(P), {k: np.concatenate(v) for k, v in pt.items() if len(v) == len(paths)}


if __name__ == "__main__":
    d = read_vtu(sys.argv[1])
    print({k: v.shape for k, v in d["point"].items()})
