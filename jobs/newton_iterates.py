#!/usr/bin/env python3
"""The Newton iterates of a running (or finished) sweep point as state files, from the run cache: every accepted
iterate y_k is evaluated (F = Phi(y_k) - y_k), and that run's output Phi(y_k), with all its diagnostics (velocity,
pressure, temperature loads: enough for plots/plot_section.py, plot_melt.py, sweep_hydrology.py), is a file in
<work>/cache written seconds before the log line "it k rms(F) = ...". Each such file is saved as a state on REF's mesh
(REF: any state of that mesh, e.g. the segment's previous point) with the patch mask of the point's d, meta sweep_d and
newton_it (the plots then show one step per iterate).

Phi(y_k) is the map's output (2000 yr on from the iterate), not y_k itself: it differs from y_k by F, i.e. by the
residual the log reports, and converges to the same steady state.

    python3 jobs/newton_iterates.py LOG.out "sweep d=781.7 km" REF.npz OUTDIR [--work work/flowline_rect_coarse]
                                    [--date 2026-11-19]

LOG: the job's slurm output; the label as in its lines (the GL round is matched too: all rounds' iterates are taken,
numbered in order). OUTDIR/iter_KK.npz.
"""
import argparse
import datetime
import glob
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ess.spectrum import patch_mask  # noqa: E402
from ess.statefile import find_name, load_state, save_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("label")
    ap.add_argument("ref")
    ap.add_argument("outdir")
    ap.add_argument("--work", default=os.path.join(ROOT, "work", "flowline_rect_coarse"))
    ap.add_argument("--date", help="the date of the log's FIRST line (YYYY-MM-DD; default: from the log's last modification, counting the "
                                   "midnights in it: the lines carry only the time of day)")
    a = ap.parse_args()
    m = re.search(r"d=([0-9.]+) km", a.label)
    if not m:
        raise SystemExit("the label must contain 'd=<km> km'")
    d = float(m.group(1)) * 1e3
    pat = re.compile(r"^(\d\d:\d\d:\d\d) " + re.escape(a.label) + r".*: it (\d+)  rms\(F\) = ([0-9.e+-]+)")
    its = []
    lines = open(a.log, errors="replace").read().splitlines()
    tre = re.compile(r"^(\d\d):(\d\d):(\d\d) ")
    tod = [int(m_.group(1)) * 3600 + int(m_.group(2)) * 60 + int(m_.group(3)) if (m_ := tre.match(l_)) else None for l_ in lines]
    days, last, offs = 0, None, []
    for t_ in tod:                                   # a day passes where the time of day falls back by over 12 h
        if t_ is not None:
            if last is not None and t_ < last - 43200:
                days += 1
            last = t_
        offs.append(days)
    if a.date:
        d0 = datetime.date.fromisoformat(a.date)
    else:
        d0 = datetime.date.fromtimestamp(os.path.getmtime(a.log)) - datetime.timedelta(days=days)
    for l_, t_, o_ in zip(lines, tod, offs):
        mm = pat.match(l_)
        if mm and t_ is not None:
            day = d0 + datetime.timedelta(days=o_)
            t = time.mktime(time.strptime(f"{day.isoformat()} {mm.group(1)}", "%Y-%m-%d %H:%M:%S"))
            its.append((t, int(mm.group(2)), float(mm.group(3))))
    print(f"log {a.log}: {days} midnight(s), first line on {d0}; {len(its)} iterate line(s) for '{a.label}'")
    if not its:
        raise SystemExit(f"no 'it k rms(F)' lines for '{a.label}' in {a.log}")
    files = [(os.path.getmtime(f), f) for f in glob.glob(os.path.join(a.work, "cache", "*.npz"))
             if not f.endswith(".tmp.npz")]
    ref = load_state(a.ref)
    n = ref.layout.size
    os.makedirs(a.outdir, exist_ok=True)
    for k, (t, it, r) in enumerate(its):
        cand = [(t - mt, f) for mt, f in files if -2.0 <= t - mt <= 120.0]
        if not cand:
            print(f"it {it} (rms {r:.3e}, {time.strftime('%H:%M:%S', time.localtime(t))}): no cache file just before "
                  f"its log line")
            continue
        _, f = min(cand)
        with np.load(f) as z:
            x = z["x"]
            if len(x) != n:
                print(f"it {it}: {os.path.basename(f)} is not on REF's mesh ({len(x)} values, REF {n})")
                continue
            diag = {kk[3:]: (z["g::" + kk[3:]], z[kk]) for kk in z.files if kk.startswith("v::")}
        frozen = dict(ref.frozen)
        kct = find_name(frozen, "ColdTempMask")
        g_ct, ct, xg = patch_mask(frozen, diag, d)
        frozen[kct] = (g_ct, ct)
        out = os.path.join(a.outdir, f"iter_{k:02d}.npz")
        save_state(out, ref.layout, x, ref.p, frozen, diag, sweep_d=d, x_gl=xg, newton_it=k, newton_rms=r)
        print(f"it {it} (rms(F) {r:.3e}): {os.path.basename(f)} -> {out}")


if __name__ == "__main__":
    main()
