#!/usr/bin/env python3
"""
The spectra of `ess spectrum` (one folder spectrum_p<Tsurf>[_e<edge>] per surface temperature) on one page:

  top      every (Tsurf, number of temperate bed columns k) tried: kept (consistent steady state) or
           discarded (hover: the reason); click a kept (green) square to show that state
  below    the selected state: basal temperature, the water sheet h (steady hydrology on the temperate
           patch), and the x-z sections of temperature, u and w with the surface drawn on them

Click a green (kept) square of the top panel to show that steady state; hover any square for its status.

Copy the folders to your laptop, then press Run (settings below) or:
    python plots/plot_spectrum.py work/land_flowline_T2k_strain "spectrum_p*_e50"
Needs numpy and plotly.
"""
import csv
import json
import glob
import os
import re
import sys
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import plot_branch_states as pbs  # noqa: E402

# ---- settings for pressing Run (paths relative to the repository)
DIR = "work/land_flowline_T2k_strain"
PATTERN = "spectrum_p*_e50,fine/spectrum_p*_e50"     # the spectrum folders (one per Tsurf); comma = several
SECTIONS = True                 # T, u, w sections (False: lines only, a much smaller html)
VEL_NZ = 40                     # vertical points of the section grids
SHOW = True
FILL_PASSED = True              # finished forcings: untried patch sizes up to fully temperate shown grey

# status codes and colours of the grid
NOT_RUN, NO_CONV, THERMAL, HYDRO, KEPT = 0, 1, 2, 3, 4
COLOURS = {NOT_RUN: "#b0b0b0", NO_CONV: "#e74c3c", THERMAL: "#f1c40f", HYDRO: "#2e86de", KEPT: "#2ca02c"}


def status(r):
    """grid code of one spectrum.csv row: not converged / forbidden by the heat equation (a cold bed node
    above melting) / forbidden by the hydrology (water sheet reaches 0) / kept"""
    if str(r.get("converged")) != "1":
        return NO_CONV
    if str(r.get("thermal_ok")) == "0":
        return THERMAL
    if str(r.get("hydro_ok")) == "0":
        return HYDRO
    return KEPT if str(r.get("kept")) == "1" else HYDRO


def read_spectra(d, pattern):
    """pattern: one glob, or several separated by commas (e.g. "spectrum_p*_e50,fine/spectrum_p*_e50");
    a Tsurf found twice keeps the folder with more rows"""
    out = {}
    files = sorted(f for pat in pattern.split(",") for f in glob.glob(os.path.join(d, pat.strip())))
    for f in files:
        m = re.search(r"spectrum_p([0-9.]+)", os.path.basename(f))
        csvf = os.path.join(f, "spectrum.csv")
        if not m or not os.path.exists(csvf):
            continue
        with open(csvf) as fh:
            rows = list(csv.DictReader(fh))
        p = float(m.group(1))
        if p not in out or len(rows) > len(out[p][2]):
            out[p] = (p, f, rows)
    return sorted(out.values())


def n_columns(f):
    """number of glaciated bed columns (patch size of a fully temperate bed) of one spectrum folder"""
    mf = os.path.join(f, "spectrum_meta.json")
    if os.path.exists(mf):
        with open(mf) as fh:
            return int(json.load(fh)["n_columns"])
    s0 = next((g for g in (os.path.join(f, "k0_state.npz"), os.path.join(f, "points", "state_0000.npz"))
               if os.path.exists(g)), None)
    if s0 is None:
        return None
    from ess.spectrum import bed_columns
    from ess.statefile import load_state
    m = re.search(r"_e([0-9.]+)$", os.path.basename(f))
    st = load_state(s0)
    try:
        _, _, ux = bed_columns(st.layout, st.frozen, st.diag, x=st.x,
                               edge_thickness=float(m.group(1)) if m else None)
    except Exception:
        return None
    return len(ux)


def stopped(rows, n=3):
    """the spectrum stopped after n consecutive failures (Newton or hydrology)"""
    tail = rows[-n:]
    return len(tail) == n and all(str(r.get("converged")) != "1" or str(r.get("hydro_ok")) == "0" for r in tail)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DIR
    pattern = sys.argv[2] if len(sys.argv) > 2 else PATTERN
    d = d if os.path.isabs(d) else os.path.join(REPO, d)
    spec = read_spectra(d, pattern)
    if not spec:
        raise SystemExit(f"no {pattern}/spectrum.csv under {d}")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pbs.VEL_PLOT = "contour" if SECTIONS else "lines"
    pbs.VEL_NZ = VEL_NZ
    P = [p for p, _, _ in spec]
    K = sorted({int(r["k"]) for _, _, rows in spec for r in rows})
    ip_of = {p: i for i, p in enumerate(P)}
    ik_of = {k: i for i, k in enumerate(K)}
    # ---- status grid and the kept states
    Z = np.full((len(K), len(P)), np.nan)
    notes = [["not tried"] * len(P) for _ in K]
    S = {}
    for p, f, rows in spec:
        for r in rows:
            k = int(r["k"])
            ik, ip = ik_of[k], ip_of[p]
            kept = str(r.get("kept")) == "1"
            Z[ik, ip] = status(r)
            xb = r.get("x_boundary_km", "")
            notes[ik][ip] = (("kept" if kept else "discarded: " + (r.get("note") or "?"))
                             + (f"<br>boundary {float(xb):.2f} km" if xb not in ("", "nan") else "")
                             + (f"<br>h at boundary {float(r['h_boundary']):.3g} m"
                                if r.get("h_boundary") not in (None, "", "nan") else ""))
            sf = os.path.join(f, "points", f"state_{k:04d}.npz")
            if kept and os.path.exists(sf):
                S[(ip, ik)] = pbs.extract(sf)
    # forcings that are finished (a warmer one has started, or the spectrum stopped after its hydrology
    # failures): every patch size up to fully temperate that was not tried counts as discarded
    ncols = [n_columns(f) for _, f, _ in spec]
    if FILL_PASSED and any(n is not None for n in ncols):
        kmax = max(n for n in ncols if n is not None)
        K_old = K
        K = sorted(set(K) | set(range(0, kmax + 1)))
        ik_of = {k: i for i, k in enumerate(K)}
        Z2 = np.full((len(K), len(P)), np.nan)
        notes2 = [["not tried"] * len(P) for _ in K]
        for p, f, rows in spec:
            ip = ip_of[p]
            for r in rows:
                ik = ik_of[int(r["k"])]
                Z2[ik, ip] = status(r)
        for ik_old, k in enumerate(K_old):
            for ip in range(len(P)):
                notes2[ik_of[k]][ip] = notes[ik_old][ip]
        for i, (p, f, rows) in enumerate(spec):
            later = any(rows2 for p2, _, rows2 in spec if p2 > p)
            if not (later or stopped(rows)) or ncols[i] is None:
                continue
            for k in range(0, ncols[i] + 1):
                ik = ik_of[k]
                if np.isnan(Z2[ik, i]):
                    Z2[ik, i] = NOT_RUN              # grey: not run, the forcing is finished
                    notes2[ik][i] = "not run: the spectrum moved on (Newton / hydrology failures)"
        S = {(ip, ik_of[K_old[ik]]): v for (ip, ik), v in S.items()}
        Z, notes = Z2, notes2
    print(f"{len(P)} surface temperatures, k = {K[0]}..{K[-1]}, {len(S)} kept states")
    if not S:
        raise SystemExit("no kept states to show")
    SL = list(S.values())
    # ---- fixed ranges over all kept states
    xmax = max(s["xb"].max() for s in SL) / 1e3
    tb = [min(s["Tb"].min() for s in SL) - 0.2, max(273.3, max(s["Tb"].max() for s in SL) + 0.1)]
    hw = [s["hw"] for s in SL if "hw" in s and np.isfinite(s["hw"]).any()]
    hr = [0.0, max(float(np.nanmax(h)) for h in hw) * 1.05] if hw else [0, 1]
    cr = {}
    if SECTIONS:
        for k_, cs in (("TS", "RdYlBu_r"), ("U", "Viridis"), ("W", "RdBu_r")):
            have = [s[k_][2] for s in SL if k_ in s]
            if not have:
                continue
            lo, hi = min(np.nanmin(g) for g in have), max(np.nanmax(g) for g in have)
            if k_ == "W":
                hi = max(abs(lo), abs(hi)) or 1.0
                lo = -hi
            cr[k_] = dict(colorscale=cs, zmin=lo, zmax=hi)

    titles = ["Tsurf vs temperate bed columns k: green kept (click), red not converged, yellow forbidden by the heat equation, blue forbidden by the hydrology, grey not run",
              "basal temperature (K)", "water sheet thickness h (m)"]
    sec = [("TS", "temperature (K)", "K"), ("U", "along-flow velocity u (m/yr)", "m/yr"),
           ("W", "vertical velocity w (m/yr)", "m/yr")] if SECTIONS else []
    sec = [t for t in sec if t[0] in cr]
    titles += [t[1] for t in sec]
    rows = len(titles)
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.05, subplot_titles=titles,
                        row_heights=[1.3] + [1] * (rows - 1))
    cb = lambda r, t: dict(title=t, len=0.8 / rows, y=1 - (r - 0.5) / rows * (1 - 0.0))
    E = go.Scatter(x=[], y=[])

    def state_traces(key):
        s = S.get(key)
        ip, ik = key
        mark = go.Scatter(x=[P[ip]], y=[K[ik]], mode="markers", hoverinfo="skip",
                          marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="black", width=3)))
        if s is None:
            return [E, E] + [E, E] * len(sec) + [mark]
        tr = [go.Scatter(x=s["xb"] / 1e3, y=np.round(s["Tb"], 4), mode="lines", line=dict(color="#c0392b"),
                         hovertemplate="x %{x:.2f} km<br>T %{y:.3f} K<extra></extra>"),
              go.Scatter(x=s["xb"] / 1e3, y=s["hw"], mode="lines+markers", line=dict(color="#2980b9"),
                         marker=dict(size=4), hovertemplate="x %{x:.2f} km<br>h %{y:.4g} m<extra></extra>")
              if "hw" in s else E]
        for i, (k_, _, unit) in enumerate(sec):
            if k_ in s:
                xg, zg, G = s[k_]
                tr.append(go.Contour(x=xg, y=zg, z=G, connectgaps=False, ncontours=30, showscale=True,
                                     contours=dict(coloring="fill", showlines=False), line=dict(width=0),
                                     colorbar=cb(4 + i, unit), **cr[k_],
                                     hovertemplate=f"x %{{x:.2f}} km<br>z %{{y:.0f}} m<br>%{{z:.4g}} {unit}"
                                                   "<extra></extra>"))
            else:
                tr.append(E)
            tr.append(go.Scatter(x=s["xs"] / 1e3, y=s["zs"], mode="lines", line=dict(color="black", width=1.5),
                                 hoverinfo="skip") if "xs" in s else E)
        tr.append(mark)
        return tr

    where = [2, 3] + [r for i in range(len(sec)) for r in (4 + i, 4 + i)] + [1]
    first = next(iter(S))
    for t, r in zip(state_traces(first), where):
        fig.add_trace(t, r, 1)
    nanim = len(where)
    # the status grid (static) and its click targets
    cs = []
    for c in range(5):                                       # discrete: one colour per code 0..4
        cs += [[c / 5, COLOURS[c]], [(c + 1) / 5 - 1e-9, COLOURS[c]]]
    cs[-1][0] = 1.0
    fig.add_trace(go.Heatmap(x=P, y=K, z=Z, colorscale=cs, zmin=-0.5, zmax=4.5, showscale=False,
                             xgap=2, ygap=1 if len(K) <= 40 else 0, text=notes,
                             customdata=[[[ip, ik] for ip in range(len(P))] for ik in range(len(K))],
                             hovertemplate="Tsurf %{x:g} K, k = %{y}<br>%{text}<extra></extra>"), 1, 1)

    def title(key):
        ip, ik = key
        s = S.get(key)
        head = f"Tsurf = {P[ip]:g} K, k = {K[ik]} temperate bed columns"
        return head + ("" if s else f": {notes[ik][ip].replace('<br>', ', ')}")

    # one frame per kept state (only those can be clicked)
    fig.frames = [go.Frame(name=f"{ip}_{ik}", data=state_traces((ip, ik)), traces=list(range(nanim)),
                           layout=dict(title_text=title((ip, ik)))) for (ip, ik) in S]
    fig.update_layout(
        title=dict(text=title(first), font=dict(size=13)), height=330 * rows, template="plotly_white",
        showlegend=False, margin=dict(t=90, b=60))
    fig.update_xaxes(title_text="Tsurf (K)", row=1, col=1)
    fig.update_yaxes(title_text="k", row=1, col=1)
    for r in range(2, rows + 1):
        fig.update_xaxes(range=[0, xmax], row=r, col=1)
    fig.update_xaxes(title_text="x (km)", row=rows, col=1)
    fig.update_yaxes(range=tb, title_text="K", row=2, col=1)
    fig.update_yaxes(range=hr, title_text="m", row=3, col=1)
    kept_keys = json.dumps([f"{ip}_{ik}" for (ip, ik) in S])
    js = """
var gd = document.getElementById('{plot_id}');
var kept = new Set(KEPT_KEYS);
gd.on('plotly_click', function (ev) {
  var c = ev.points[0].customdata;
  if (!Array.isArray(c)) return;
  var name = c[0] + '_' + c[1];
  if (!kept.has(name)) return;            // discarded / not tried: nothing to show
  Plotly.animate(gd, [name], {mode: 'immediate', frame: {duration: 0, redraw: true}, transition: {duration: 0}});
});
""".replace("KEPT_KEYS", kept_keys)
    out = os.path.join(d, "spectra.html")
    fig.write_html(out, include_plotlyjs="cdn", post_script=js, auto_play=False)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    if SHOW:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
