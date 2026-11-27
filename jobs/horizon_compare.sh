#!/bin/bash
# Short-horizon comparison: the main sweep (map 10 steps = 500 yr) against the branches off it (jobs/gl_sweep_horizon.sh: 250 yr = 5
# steps, 100 yr = 2 steps), point by point from the logs slurms/gl_sweep_<id>.out: per grounding-line distance d the Newton iterations, the
# model runs, the last rms(F) and the time the point took (from the previous point's last line, the first from the start line of the
# log). The logs compared are those started from the main sweep's points state_04989 (the main sweep) and state_03989 (the branches),
# or the job ids given.
#   cd /SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates && bash jobs/horizon_compare.sh
#   bash jobs/horizon_compare.sh 32436 32470 32471       # chosen logs
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import glob, re, sys

hdr = re.compile(r"^(\d\d):(\d\d):(\d\d) sweep on \S+ from (\S+\.npz).*?map (\d+) steps")
pat = re.compile(r"^(\d\d):(\d\d):(\d\d) sweep d=([\d.]+) km \(GL round (\d+)\): it (\d+)\s+rms\(F\) = ([\d.e+-]+).*?model runs so far (\d+)")
ids = sys.argv[1:]
files = [f"slurms/gl_sweep_{i}.out" for i in ids] if ids else sorted(glob.glob("slurms/gl_sweep_*.out"))
logs = []
for f in files:
    day, last, h0, start = 0, None, None, None
    pts, order = {}, []
    try:
        lines = open(f, errors="replace").read().splitlines()
    except OSError:
        print("no", f); continue
    for ln in lines:
        m = hdr.match(ln)
        if m and start is None:
            start = (m.group(4), int(m.group(5)))
            h0 = None
        t = None
        m2 = pat.match(ln)
        m1 = hdr.match(ln)
        mm = m2 or m1
        if mm:
            t = int(mm.group(1)) * 3600 + int(mm.group(2)) * 60 + int(mm.group(3))
            if last is not None and t + day * 86400 < last:
                day += 1
            t += day * 86400
            last = t
            if h0 is None:
                h0 = t
        if m2:
            d = m2.group(4)
            if d not in pts:
                pts[d] = dict(its=0, runs=0, rms=None, t=t, first=t, rounds=0)
                order.append(d)
            p = pts[d]
            p.update(its=max(p["its"], int(m2.group(6))), runs=max(p["runs"], int(m2.group(8))), rms=float(m2.group(7)), t=t,
                     rounds=max(p["rounds"], int(m2.group(5))))
    if start is None or not pts:
        continue
    keep = bool(ids) or "state_04989" in start[0] or "state_03989" in start[0]
    if not keep:
        continue
    prev = h0
    for d in order:
        pts[d]["elapsed"] = pts[d]["t"] - prev
        prev = pts[d]["t"]
    logs.append((f.split("_")[-1][:-4], start[1], order, pts))
if not logs:
    sys.exit("no sweep logs found")
fm = lambda s: f"{s // 3600}:{s % 3600 // 60:02d}"
print(f"{'job':>7} {'map':>4}  per point: d km | Newton its | model runs | last rms(F) | time (last point of a log may still be running)")
for jid, ms, order, pts in logs:
    print(f"\n{jid:>7} {ms * 50:>3} yr ({ms} steps)")
    for d in order:
        p = pts[d]
        print(f"   d {d:>6} | its {p['its']:>2} | runs {p['runs']:>3} | rms(F) {p['rms']:.2e} | {fm(p['elapsed'])} h")
common = sorted(set.intersection(*[set(o) for _, _, o, _ in logs]), key=float, reverse=True)
if common and len(logs) > 1:
    print("\nside by side, points every log has reached (runs / time):")
    print("  d km   " + "".join(f"{jid + ' (' + str(ms * 50) + ' yr)':>26}" for jid, ms, _, _ in logs))
    for d in common:
        print(f"  {d:>6} " + "".join(f"{str(pts[d]['runs']) + ' runs, ' + fm(pts[d]['elapsed']) + ' h':>26}" for _, _, _, pts in logs))
PY
