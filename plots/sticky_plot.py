#!/usr/bin/env python3
"""Plot a sticky-air lava run (sticky/drop.sif): Elmer's VTU output (one .pvtu per output step, a .vtu piece per rank)
read with numpy alone (ascii, base64 binary or raw appended data). One page, a slider over the output steps:
  left : the section along the drop's centre line (y = --y, default 4 m): the lava fraction over x and z (nodes within
         half a cell of y), the 0.5 contour drawn in;
  right: the lava thickness from above, h(x, y) = sum over the column of the fraction x its layer thickness.

    python3 plots/sticky_plot.py RUN_DIR [--y 4.0] [--out RUN_DIR/sticky.html]

RUN_DIR: the run's folder (sticky/run_<job>), searched for *.pvtu (or *.vtu if there are none)."""
import base64
import glob
import os
import re
import struct
import sys
import zlib

import numpy as np

DT = {"Float64": "<f8", "Float32": "<f4", "Int32": "<i4", "Int64": "<i8", "UInt32": "<u4", "UInt64": "<u8",
      "Int8": "<i1", "UInt8": "<u1"}


def read_vtu(path):
    """points (n, 3) and the point data {lower-case name: array (n,) or (n, k)}"""
    raw = open(path, "rb").read()
    head = raw.split(b"<AppendedData", 1)[0].decode("latin-1")
    hdr = "<u8" if 'header_type="UInt64"' in head else "<u4"
    hsize = np.dtype(hdr).itemsize
    compressed = "vtkZLibDataCompressor" in head
    app = None
    if b"<AppendedData" in raw:
        i = raw.index(b"<AppendedData")
        i = raw.index(b">", i) + 1
        i = raw.index(b"_", i) + 1
        app = raw[i:]
        if b'encoding="base64"' in raw[raw.index(b"<AppendedData"):raw.index(b"<AppendedData") + 200]:
            app = base64.b64decode(app.split(b"</AppendedData")[0].strip())

    def block(buf, off):
        n = int(np.frombuffer(buf[off:off + hsize], hdr)[0])
        return buf[off + hsize: off + hsize + n]

    def parse(tag, body):
        typ = re.search(r'type="(\w+)"', tag).group(1)
        dt = DT[typ]
        nc = int((re.search(r'NumberOfComponents="(\d+)"', tag) or [0, 1])[1]) if "NumberOfComponents" in tag else 1
        fmt = re.search(r'format="(\w+)"', tag).group(1)
        if fmt == "ascii":
            a = np.array(body.split(), dtype=dt)
        elif fmt == "binary":
            b = base64.b64decode(body.strip())
            if compressed:
                raise SystemExit("compressed binary VTU not supported")
            a = np.frombuffer(b[hsize:], dt)
        else:   # appended
            off = int(re.search(r'offset="(\d+)"', tag).group(1))
            if compressed:
                raise SystemExit("compressed appended VTU not supported")
            a = np.frombuffer(block(app, off), dt)
        return a.reshape(-1, nc) if nc > 1 else a

    txt = raw.decode("latin-1")
    pts = None
    data = {}
    for m in re.finditer(r"<DataArray([^>]*)>(.*?)</DataArray>|<DataArray([^>]*)/>", txt, re.S):
        tag = m.group(1) if m.group(1) is not None else m.group(3)
        body = m.group(2) or ""
        name = (re.search(r'Name="([^"]*)"', tag) or [None, ""])[1]
        sec_start = txt.rfind("<", 0, m.start())
        # which section is this array in?
        before = txt[:m.start()]
        in_points = before.rfind("<Points") > max(before.rfind("</Points"), before.rfind("<PointData"), before.rfind("<Cells"))
        in_pdata = before.rfind("<PointData") > before.rfind("</PointData")
        if in_points:
            pts = parse(tag, body).reshape(-1, 3)
        elif in_pdata and name:
            data[name.lower()] = parse(tag, body)
    return pts, data


def read_step(pvtu):
    if pvtu.endswith(".vtu"):
        return read_vtu(pvtu)
    d = os.path.dirname(pvtu)
    src = re.findall(r'<Piece[^>]*Source="([^"]+)"', open(pvtu).read())
    P, D = [], {}
    for s in src:
        p, dd = read_vtu(os.path.join(d, s))
        P.append(p)
        for k, v in dd.items():
            D.setdefault(k, []).append(v)
    return np.vstack(P), {k: np.concatenate(v) for k, v in D.items()}


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    run = args[0]
    yc = float(args[args.index("--y") + 1]) if "--y" in args else 4.0
    out = args[args.index("--out") + 1] if "--out" in args else os.path.join(run, "sticky.html")
    files = sorted(glob.glob(os.path.join(run, "**", "*.pvtu"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(run, "**", "*.vtu"), recursive=True))
    if not files:
        raise SystemExit(f"no .pvtu / .vtu under {run}")
    print(f"{len(files)} output steps: {os.path.basename(files[0])} .. {os.path.basename(files[-1])}")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    frames, labels = [], []
    for k, f in enumerate(files):
        P, D = read_step(f)
        lav = D.get("lava")
        if lav is None:
            raise SystemExit(f"no 'lava' in {f}: {sorted(D)}")
        # merge duplicate nodes (shared between pieces)
        key = np.round(P / 1e-6).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        P, lav = P[idx], lav[idx]
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        xs, ys, zs = np.unique(np.round(x, 6)), np.unique(np.round(y, 6)), np.unique(np.round(z, 6))
        # section: the footprint row nearest yc
        yr = ys[np.argmin(abs(ys - yc))]
        m = abs(y - yr) < 1e-6
        ix = np.searchsorted(xs, np.round(x[m], 6)); iz = np.searchsorted(zs, np.round(z[m], 6))
        S = np.full((len(zs), len(xs)), np.nan); S[iz, ix] = lav[m]
        # thickness: sum of the fraction x the layer thickness (trapezoid along each column)
        F = np.zeros((len(ys), len(xs), len(zs)))
        F[np.searchsorted(ys, np.round(y, 6)), np.searchsorted(xs, np.round(x, 6)), np.searchsorted(zs, np.round(z, 6))] = np.clip(lav, 0, 1)
        H = (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(F, zs, axis=2)
        frames.append((xs, zs, S, ys, H))
        t = re.search(r"(\d+)\.p?vtu$", f)
        labels.append(t.group(1) if t else str(k))
        print(f"  {os.path.basename(f)}: {len(P)} nodes, lava max {np.nanmax(lav):.4f} min {np.nanmin(lav):.2e}, "
              f"volume {H.sum() * np.mean(np.diff(xs)) * np.mean(np.diff(ys)):.4f} m^3 (approx.)")
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        subplot_titles=(f"section at y = {yc:g} m: lava fraction", "lava thickness (m)"))
    hmax = max(np.nanmax(fr[4]) for fr in frames)
    traces, per = [], 3
    for k, (xs, zs, S, ys, H) in enumerate(frames):
        v = (k == 0)
        traces += [go.Heatmap(x=xs, y=zs, z=S, zmin=0, zmax=1, colorscale="Inferno", visible=v,
                              colorbar=dict(title="f", x=0.56, len=0.9)),
                   go.Contour(x=xs, y=zs, z=S, contours=dict(start=0.5, end=0.5, coloring="none"), showscale=False,
                              line=dict(color="cyan", width=1.5), visible=v),
                   go.Heatmap(x=xs, y=ys, z=H, zmin=0, zmax=hmax, colorscale="Viridis", visible=v,
                              colorbar=dict(title="h (m)", x=1.0, len=0.9))]
    for i, t in enumerate(traces):
        fig.add_trace(t, row=1, col=1 if i % per < 2 else 2)
    steps = []
    for k, lab in enumerate(labels):
        vis = [False] * len(traces)
        vis[k * per:(k + 1) * per] = [True] * per
        steps.append(dict(method="update", args=[{"visible": vis}], label=lab))
    fig.update_layout(sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="output "))], template="plotly_white",
                      height=650)
    fig.update_yaxes(title="z (m)", row=1, col=1)
    fig.update_xaxes(title="x (m, down the slope)", row=1, col=1)
    fig.update_xaxes(title="x (m)", row=1, col=2)
    fig.update_yaxes(title="y (m)", scaleanchor="x2", row=1, col=2)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
