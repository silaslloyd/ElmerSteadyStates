#!/usr/bin/env python3
"""
CPU time of the land strip's FSSA runs over the full 10,000 years against the time step (plotly): three lines, full FSSA,
implicit (adapted FSSA, 8 coupled iterations, tolerance 1e-4) with BDF 1 and with BDF 2. The CPU time is slurm's allocated
core-seconds of the finished job (CPUTimeRAW = elapsed x cores, shown in core-hours); of several jobs for one (line, step)
the last COMPLETED one. Run on tethys (it asks sacct):

    python3 plots/fssa_cpu_time.py [--out FILE.html] [--since 2026-11-01] [--sacct-file FILE]

Job names (jobs/land3d_log100_*.sh and the earlier series):
    full FSSA           log100-dt<N>                                  (N: the step in yr)
    implicit BDF 1      log100-impt-dt<N>   or  log100-impt-dt<N>-bdf1-ilu3
    implicit BDF 2      log100-impt-dt<N>-bdf2-ilu3
--sacct-file: a saved `sacct -u $USER -S DATE -X -n -P -o JobID,JobName,State,Elapsed,AllocCPUS,CPUTimeRAW` (| separated), for
trying the page elsewhere. Jobs that did not finish (cancelled, failed, running) are not plotted.
"""
import re
import subprocess
import sys


def read_jobs(since, path):
    if path:
        txt = open(path).read()
    else:
        txt = subprocess.run(["sacct", "-u", __import__("os").environ.get("USER", ""), "-S", since, "-X", "-n", "-P", "-o",
                              "JobID,JobName,State,Elapsed,AllocCPUS,CPUTimeRAW"], capture_output=True, text=True, check=True).stdout
    rows = []
    for ln in txt.splitlines():
        f = ln.strip().split("|")
        if len(f) < 6:
            continue
        try:
            rows.append(dict(id=int(f[0]), name=f[1], state=f[2], elapsed=f[3], cpus=int(f[4]), cpusec=float(f[5])))
        except ValueError:
            continue
    return rows


def classify(name):
    """(line, dt) of a job name, or None"""
    m = re.fullmatch(r"log100-dt(\d+)", name)
    if m:
        return "full FSSA", int(m.group(1))
    m = re.fullmatch(r"log100-impt-dt(\d+)(-bdf1-ilu3)?", name)
    if m:
        return "implicit, BDF 1", int(m.group(1))
    m = re.fullmatch(r"log100-impt-dt(\d+)-bdf2-ilu3", name)
    if m:
        return "implicit, BDF 2", int(m.group(1))
    return None


def main():
    a = sys.argv[1:]
    out, since, sf = "fssa_cpu_time.html", "2026-11-01", None
    for k, v in (("--out", "out"), ("--since", "since"), ("--sacct-file", "sf")):
        if k in a:
            i = a.index(k)
            val = a[i + 1]
            del a[i:i + 2]
            if v == "out":
                out = val
            elif v == "since":
                since = val
            else:
                sf = val
    best = {}
    for r in read_jobs(since, sf):
        c = classify(r["name"])
        if c is None or not r["state"].startswith("COMPLETED"):
            continue
        if c not in best or r["id"] > best[c]["id"]:
            best[c] = r
    if not best:
        raise SystemExit("no finished jobs with the expected names (log100-dtN, log100-impt-dtN[-bdf1-ilu3], log100-impt-dtN-bdf2-ilu3)")
    import plotly.graph_objects as go
    fig = go.Figure()
    style = {"full FSSA": ("#444444", "circle"), "implicit, BDF 1": ("#1f77b4", "circle"), "implicit, BDF 2": ("#d62728", "square")}
    for line in ("full FSSA", "implicit, BDF 1", "implicit, BDF 2"):
        pts = sorted((dt, r) for (ln, dt), r in best.items() if ln == line)
        if not pts:
            continue
        col, sym = style[line]
        fig.add_trace(go.Scatter(
            x=[dt for dt, _ in pts], y=[r["cpusec"] / 3600 for _, r in pts], name=line, mode="lines+markers",
            line=dict(color=col, width=2.5), marker=dict(size=9, symbol=sym),
            customdata=[[r["id"], r["elapsed"], r["cpus"]] for _, r in pts],
            hovertemplate=line + "<br>dt %{x} yr<br>%{y:,.0f} core-hours<br>job %{customdata[0]}, elapsed %{customdata[1]}, %{customdata[2]} cores<extra></extra>"))
        print(f"{line}: " + ", ".join(f"dt {dt}: {r['cpusec'] / 3600:,.0f} core-h (job {r['id']})" for dt, r in pts))
    fig.update_layout(template="plotly_white", height=620, title="CPU time of the 10,000 yr land strip FSSA runs against the time step",
                      xaxis=dict(title="time step (yr)", type="log"), yaxis=dict(title="CPU time (core-hours)", type="log"),
                      hovermode="closest", legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"))
    fig.write_html(out, include_plotlyjs="cdn")
    print("->", out)


if __name__ == "__main__":
    main()
