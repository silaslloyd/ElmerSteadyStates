#!/usr/bin/env python3
"""Plot a sticky-air lava run (sticky/drop.sif): Elmer's VTU output (one .pvtu per output step, a .vtu piece per rank)
read with numpy alone (ascii, base64 binary or raw appended data). One page, a slider over the output steps:
  left : the section along the drop's centre line (y = --y, default 4 m): the lava fraction over x and z (nodes within
         half a cell of y), the 0.5 contour drawn in;
  right: the lava thickness from above, h(x, y) = sum over the column of the fraction x its layer thickness.

    python3 plots/sticky_plot.py RUN_DIR [--y 4.0] [--out RUN_DIR/sticky.html] [--procs N (default min(8, cores))]

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


def read_vtu(path, want=("lava",)):
    """points (n, 3) and the point data {lower-case name: array} for the names in want (all if want is None).
    Fast path for appended data (Elmer's default): only the XML header is parsed, only the wanted arrays decoded."""
    raw = open(path, "rb").read()
    ia = raw.find(b"<AppendedData")
    head = (raw[:ia] if ia >= 0 else raw).decode("latin-1")
    hdr = "<u8" if 'header_type="UInt64"' in head else "<u4"
    hsize = np.dtype(hdr).itemsize
    compressed = "vtkZLibDataCompressor" in head
    app = None
    if ia >= 0:
        i = raw.index(b">", ia) + 1
        i = raw.index(b"_", i) + 1
        app = raw[i:]
        if b'encoding="base64"' in raw[ia:ia + 200]:
            app = base64.b64decode(app.split(b"</AppendedData")[0].strip())
    # the section boundaries in the header, once
    p0, p1 = head.find("<Points"), head.find("</Points")
    d0, d1 = head.find("<PointData"), head.find("</PointData")

    def parse(tag, body):
        typ = re.search(r'type="(\w+)"', tag).group(1)
        dt = DT[typ]
        mc = re.search(r'NumberOfComponents="(\d+)"', tag)
        nc = int(mc.group(1)) if mc else 1
        fmt = re.search(r'format="(\w+)"', tag).group(1)
        if fmt == "ascii":
            a = np.array(body.split(), dtype=dt)
        elif fmt == "binary":
            bb = base64.b64decode(body.strip())
            if compressed:
                raise SystemExit("compressed binary VTU not supported")
            a = np.frombuffer(bb[hsize:], dt)
        else:   # appended
            off = int(re.search(r'offset="(\d+)"', tag).group(1))
            if compressed:
                raise SystemExit("compressed appended VTU not supported")
            n = int(np.frombuffer(app[off:off + hsize], hdr)[0])
            a = np.frombuffer(app[off + hsize: off + hsize + n], dt)
        return a.reshape(-1, nc) if nc > 1 else a

    pts, data = None, {}
    for m in re.finditer(r"<DataArray([^>]*?)(/>|>(.*?)</DataArray>)", head, re.S):
        tag, body, pos = m.group(1), m.group(3) or "", m.start()
        name = (re.search(r'Name="([^"]*)"', tag) or [None, ""])[1]
        if p0 >= 0 and p0 < pos < (p1 if p1 >= 0 else len(head)):
            pts = parse(tag, body).reshape(-1, 3)
        elif d0 >= 0 and d0 < pos < (d1 if d1 >= 0 else len(head)) and name:
            if want is None or name.lower() in want:
                data[name.lower()] = parse(tag, body)
    return pts, data


def read_step(pvtu, want=("lava",)):
    if pvtu.endswith(".vtu"):
        return read_vtu(pvtu, want)
    d = os.path.dirname(pvtu)
    src = re.findall(r'<Piece[^>]*Source="([^"]+)"', open(pvtu).read())
    P, D = [], {}
    for s in src:
        p, dd = read_vtu(os.path.join(d, s), want)
        P.append(p)
        for k, v in dd.items():
            D.setdefault(k, []).append(v)
    return np.vstack(P), {k: np.concatenate(v) for k, v in D.items()}


def process(args):
    """one output: the section at yc and the thickness map (duplicate nodes of the pieces land on the same cell)"""
    f, yc = args
    P, D = read_step(f)
    lav = D.get("lava")
    if lav is None:
        raise SystemExit(f"no 'lava' in {f}: {sorted(D)}")
    x, y, z = np.round(P[:, 0], 6), np.round(P[:, 1], 6), np.round(P[:, 2], 6)
    xs, ys, zs = np.unique(x), np.unique(y), np.unique(z)
    ix, iy, iz = np.searchsorted(xs, x), np.searchsorted(ys, y), np.searchsorted(zs, z)
    jy = np.argmin(abs(ys - yc))
    m = iy == jy
    S = np.full((len(zs), len(xs)), np.nan)
    S[iz[m], ix[m]] = lav[m]
    F = np.zeros((len(ys), len(xs), len(zs)))
    F[iy, ix, iz] = np.clip(lav, 0, 1)
    H = (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(F, zs, axis=2)
    info = (f"  {os.path.basename(f)}: {len(P)} nodes (with duplicates), lava max {np.nanmax(lav):.4f} min "
            f"{np.nanmin(lav):.2e}, volume {H.sum() * np.mean(np.diff(xs)) * np.mean(np.diff(ys)):.4f} m^3 (approx.)")
    return (xs, zs, S, ys, H), info


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
    nproc = int(args[args.index("--procs") + 1]) if "--procs" in args else min(8, os.cpu_count() or 1)
    jobs = [(f, yc) for f in files]
    if nproc > 1 and len(files) > 1:
        from multiprocessing import Pool
        with Pool(min(nproc, len(files))) as pool:
            res = pool.map(process, jobs)
    else:
        res = [process(j) for j in jobs]
    for k, (f, (fr, info)) in enumerate(zip(files, res)):
        frames.append(fr)
        t = re.search(r"(\d+)\.p?vtu$", f)
        labels.append(t.group(1) if t else str(k))
        print(info)
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
