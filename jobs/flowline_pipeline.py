#!/usr/bin/env python3
"""The whole marine flowline pipeline in one unattended job, every stage resumable:

  S0  setup + init of the config's initial state (the Schoof / SSA solution with its shelf: grounding line 686.3 km),
      the shelf break of jobs/flowline_coarse_newton.cmd                                         -> pipeline/start1.npz
  S1  Stokes + free-surface transient, temperate bed (ESS_SLIDING=temperate), grounding line by contact every step,
      in batches of --batch chunks (1 chunk = --steps x --dt = 100 x 10 yr) until the rms change per chunk is below
      --rms-tol (1e-3)                                                                          -> pipeline/transient.npz
  S2  Newton on that (Stokes only, temperate), grounding line moved by contact between solves   -> pipeline/stokes_newton.npz
  S3  the whole bed temperate (ColdTempMask 1), steady heat (--heat-steps x --heat-dt), then the full Newton with the
      ct mask fixed ([masks] ct = false)                                                          -> pipeline/steady_temperate.npz
  S4  the bed sweep (jobs/sweep_refined.py): patch length from the whole grounded bed to 0 in --step-max-km (50) km
      steps with a short Newton map (--map-steps 10 x 50 yr = 500 yr); a failed step is halved (ess sweep halves down to
      --step-min-km, then the sweep stops). Output work/<case>/<out-root>/seg_k*/.
  S5  while S4 runs, every --poll-min minutes: the converged points are checked for the first ok / bad switch between
      neighbours (jobs/hydro_bisect.py --list) for the hydrology (the sheet runs dry) and for the thermal criterion (a
      cold bed node above melting); when one exists, that bisector is submitted as its OWN job
      (jobs/flowline_hydro_bisect.cmd, CRITERION=hydro | thermal), once. At most 3 jobs of the pipeline: this one and
      the two bisectors.

Everything is run from the repository root with the corrected contact function (ESS_CONTACT_FUNCTION=elmerstreams).
Flags in work/<case>/pipeline/flags/: DONE (finished), GAVEUP (three job starts in a row without progress),
bisect_hydro.json / bisect_thermal.json (the spawned bisector's job id and number of spawns).

    python3 jobs/flowline_pipeline.py [--until s0|s1|s2|s3|s4] [--rms-tol 1e-3] [--max-chunks 200] ...
sbatch jobs/flowline_pipeline.cmd runs it and chains its own continuation when the job's time runs out.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "jobs"))
from ess.config import load_config  # noqa: E402
from ess.spectrum import gl_position  # noqa: E402
from ess.statefile import find_name, load_state, save_state  # noqa: E402
from sweep_refined import converged, write_config  # noqa: E402

PY = sys.executable
RMS_RE = re.compile(r"rms change\s+([0-9.]+(?:[eE][+-]?[0-9]+)?)")
BISECT_NAME = {"hydro": "flbis-hydro", "thermal": "flbis-thermal"}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} pipeline: {msg}", flush=True)


def run(cmd, env=None, capture_rms=False):
    """Run a command from the repository root, output streamed; returns the last 'rms change' (capture_rms) or None.
    Raises CalledProcessError on a non-zero exit."""
    log("$ " + " ".join(cmd))
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.Popen(cmd, cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = None
    for line in p.stdout:
        print(line, end="", flush=True)
        m = RMS_RE.search(line)
        if m:
            last = float(m.group(1))
    if p.wait() != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return last


def sh(cmd, timeout=600):
    """A short command's output (empty string and no exception if it fails)"""
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except Exception as ex:                                  # noqa: BLE001
        return f"ERROR {ex}"


def npoints(root):
    return sum(len(converged(os.path.dirname(f))) for f in glob.glob(os.path.join(root, "seg_*", "sweep.csv")))


class Pipeline:
    def __init__(self, a):
        self.a = a
        self.cfg = load_config(a.config)
        self.work = self.cfg["_work"]
        self.pdir = os.path.join(self.work, "pipeline")
        self.flags = os.path.join(self.pdir, "flags")
        os.makedirs(self.flags, exist_ok=True)
        j = lambda f: os.path.join(self.pdir, f)             # noqa: E731
        self.f = dict(state0=j("state0.npz"), start1=j("start1.npz"), trans=j("transient.npz"),
                      newton1=j("stokes_newton.npz"), temp=j("start_temperate.npz"), heat=j("heat.npz"),
                      steady=j("steady_temperate.npz"))
        self.cfgp = os.path.join(self.work, "refined", "pipe0.toml")
        self.sweep_root = os.path.join(self.work, a.out_root)
        self.stop = threading.Event()

    # ------------------------------------------------------------------ flags
    def flag(self, name):
        return os.path.join(self.flags, name)

    def signature(self):
        have = sorted(k for k, v in self.f.items() if os.path.exists(v))
        t = os.path.getmtime(self.f["trans"]) if os.path.exists(self.f["trans"]) else 0
        return {"files": have, "points": npoints(self.sweep_root), "trans_mtime": int(t // 600)}

    def check_progress(self):
        """GAVEUP after --patience job starts in a row with the same signature (no stage done, no new point)"""
        fn = self.flag("progress.json")
        sig = self.signature()
        prev = json.load(open(fn)) if os.path.exists(fn) else {"sig": None, "same": 0}
        same = prev["same"] + 1 if prev["sig"] == sig else 0
        json.dump({"sig": sig, "same": same}, open(fn, "w"))
        if same >= self.a.patience:
            log(f"no progress in {same + 1} job starts in a row: giving up (flags/GAVEUP)")
            open(self.flag("GAVEUP"), "w").write(time.strftime("%F %T\n"))
            return False
        return True

    # ------------------------------------------------------------------ stages
    def s0(self):
        if not os.path.exists(self.f["state0"]):
            base = self.cfg["case"]["res_points"]
            os.makedirs(os.path.dirname(self.cfgp), exist_ok=True)
            write_config(self.a.config, self.cfgp, base, "{work}/pipe0")
            run([PY, "-u", "-m", "ess", "setup", self.cfgp])
            run([PY, "-u", "-m", "ess", "init", self.cfgp, "--out", self.f["state0"]])
        if not os.path.exists(self.f["start1"]):
            run([PY, "-u", "jobs/shelf_break.py", self.f["state0"], self.f["start1"], "--strip-divide-r-km", "1520.8",
                 "--r-break-km", "450", "--factor", "30", "--width-km", "10", "--force"])
        s = load_state(self.f["start1"])
        log(f"S0 done: initial state, grounding line {gl_position(s.frozen, s.diag) / 1e3:.2f} km")

    def s1(self):
        if os.path.exists(self.flag("s1_done")):
            return
        a = self.a
        cnt_f = self.flag("s1_chunks")
        chunks = int(open(cnt_f).read()) if os.path.exists(cnt_f) else 0
        while chunks < a.max_chunks:
            cur = self.f["trans"] if os.path.exists(self.f["trans"]) else self.f["start1"]
            s_b = load_state(cur)
            gl_before = gl_position(s_b.frozen, s_b.diag) / 1e3
            rms = run([PY, "-u", "-m", "ess", "run", self.cfgp, "--state", cur, "--stokes-only", "--gl", "contact",
                       "--dt", f"{a.dt:g}", "--steps", str(a.steps), "--chunks", str(a.batch), "--save-every", "5",
                       "--chunk-offset", str(chunks), "--linear-stokes", "direct", "--out", self.f["trans"]],
                      env={"ESS_SLIDING": "temperate"}, capture_rms=True)
            chunks += a.batch
            open(cnt_f, "w").write(str(chunks))
            s = load_state(self.f["trans"])
            log(f"S1: {chunks} chunks ({chunks * a.steps * a.dt:g} yr), rms change {rms}, grounding line "
                f"{gl_position(s.frozen, s.diag) / 1e3:.2f} km")
            gl_now = gl_position(s.frozen, s.diag) / 1e3
            if rms is not None and rms < a.rms_tol and abs(gl_now - gl_before) <= a.gl_tol_km:
                break
            if rms is not None and rms < a.rms_tol:
                log(f"S1: rms change below {a.rms_tol:g} but the grounding line moved {gl_now - gl_before:+.2f} km in "
                    f"this batch (> {a.gl_tol_km:g} km): carrying on")
        else:
            log(f"S1: not below {a.rms_tol:g} in {a.max_chunks} chunks: carrying on with the state as it is")
        open(self.flag("s1_done"), "w").write(time.strftime("%F %T\n"))

    def s2(self):
        if os.path.exists(self.f["newton1"]):
            return
        try:
            run([PY, "-u", "-m", "ess", "newton", self.cfgp, "--state", self.f["trans"], "--stokes-only",
                 "--out", self.f["newton1"]], env={"ESS_SLIDING": "temperate"})
        except subprocess.CalledProcessError:
            log("S2: Newton (Stokes only) did not converge: carrying on from the transient's state")
            if not os.path.exists(self.f["newton1"]):
                s = load_state(self.f["trans"])
                save_state(self.f["newton1"], s.layout, s.x, s.p, s.frozen, s.diag, **s.meta)

    def s3(self):
        if os.path.exists(self.f["steady"]):
            return
        if not os.path.exists(self.f["temp"]):
            s = load_state(self.f["newton1"])
            k = find_name(s.frozen, "ColdTempMask")
            g, v = s.frozen[k]
            log(f"S3: ColdTempMask {int((v > 0).sum())} of {len(v)} nodes temperate; all temperate now")
            import numpy as np
            s.frozen[k] = (g, np.ones_like(v))
            save_state(self.f["temp"], s.layout, s.x, s.p, s.frozen, s.diag, **s.meta)
        if not os.path.exists(self.f["heat"]):
            run([PY, "-u", "-m", "ess", "run", self.cfgp, "--state", self.f["temp"], "--heat-only",
                 "--dt", f"{self.a.heat_dt:g}", "--steps", str(self.a.heat_steps), "--gl", "frozen", "--no-remask",
                 "--out", self.f["heat"]])
        run([PY, "-u", "-m", "ess", "newton", self.cfgp, "--state", self.f["heat"], "--out", self.f["steady"]])
        s = load_state(self.f["steady"])
        log(f"S3 done: steady temperate state, grounding line {gl_position(s.frozen, s.diag) / 1e3:.2f} km")

    def s4(self):
        a = self.a
        s = load_state(self.f["steady"])
        d0 = gl_position(s.frozen, s.diag) / 1e3                 # the whole grounded bed temperate
        env = {"ESS_NEWTON_TOL": os.environ.get("NEWTON_TOL", "5e-5")}
        cmd = [PY, "-u", "jobs/sweep_refined.py", a.config, "--from-state", self.f["steady"], "--d-km", f"{d0:g}",
               "--d-end-km", "0", "--span-km", "50", "--pad-km", "10", "--fine", "250", "--step-frac", "1.0",
               "--step-min-km", f"{a.step_min_km:g}", "--step-max-km", f"{a.step_max_km:g}",
               "--map-steps", str(a.map_steps), "--out-root", a.out_root, "--tag-prefix", "k", "--predictor", "none"]
        th = threading.Thread(target=self.monitor, daemon=True)
        th.start()
        fails = 0
        try:
            while True:
                before = npoints(self.sweep_root)
                try:
                    run(cmd, env=env)
                    log("S4: the sweep ended (finished, or stopped at the smallest step)")
                    break
                except subprocess.CalledProcessError:
                    after = npoints(self.sweep_root)
                    fails = 0 if after > before else fails + 1
                    log(f"S4: the sweep exited with an error ({after - before} new points), failures in a row: {fails}")
                    if fails >= 3:
                        log("S4: three failures in a row without a new point: stopping the sweep")
                        break
                    time.sleep(60)
        finally:
            self.stop.set()
        self.check_bisectors(final=True)

    # ------------------------------------------------------------------ bisectors
    def queue_names(self):
        out = sh(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"], timeout=60)
        if out.startswith("ERROR"):
            return None
        return [x.strip() for x in out.splitlines() if x.strip()]

    def bisector_done(self, info):
        """the spawned job's log says 'done: the ... status switches' when the bisection finished"""
        f = os.path.join(ROOT, "slurms", f"flowline_hydro_bisect_{info.get('job')}.out")
        return os.path.exists(f) and "done: the" in open(f, errors="replace").read()

    def check_bisectors(self, final=False):
        a = self.a
        names = self.queue_names()
        if names is None:
            log("squeue not available: no bisector spawned this time")
            return
        for crit in ("hydro", "thermal"):
            fn = self.flag(f"bisect_{crit}.json")
            info = json.load(open(fn)) if os.path.exists(fn) else {"job": None, "spawns": 0}
            name = BISECT_NAME[crit]
            if name in names:
                continue                                       # running or pending: leave it
            if info["job"] and self.bisector_done(info):
                continue                                       # finished
            if info["spawns"] >= a.max_spawns:
                continue
            out = sh([PY, "jobs/hydro_bisect.py", a.config, self.sweep_root, "--criterion", crit, "--list",
                      "--map-steps", str(a.map_steps)])
            if not re.search(r"bracket: \(", out):
                log(f"{crit}: no ok / bad switch between neighbouring converged points yet")
                continue
            # the 3-job cap: this job + at most one bisector of each kind
            r = sh(["sbatch", "--parsable", "-J", name,
                    f"--export=ALL,ROOTS={self.sweep_root},CRITERION={crit},MAP_STEPS={a.map_steps},CONFIG={a.config}",
                    "jobs/flowline_hydro_bisect.cmd"], timeout=120)
            m = re.match(r"\s*(\d+)", r)
            if not m:
                log(f"{crit}: sbatch failed: {r.strip()}")
                continue
            info = {"job": int(m.group(1)), "spawns": info["spawns"] + 1, "time": time.strftime("%F %T")}
            json.dump(info, open(fn, "w"))
            log(f"{crit}: a bracket exists: bisector submitted as job {info['job']} ({name}, spawn {info['spawns']})")

    def monitor(self):
        while not self.stop.wait(self.a.poll_min * 60):
            try:
                self.check_bisectors()
            except Exception as ex:                            # noqa: BLE001
                log(f"monitor: {ex!r} (carrying on)")

    # ------------------------------------------------------------------ main
    def main(self):
        a = self.a
        os.environ.setdefault("ESS_CONTACT_FUNCTION", "elmerstreams")
        log(f"contact function {os.environ['ESS_CONTACT_FUNCTION']}, case {a.config}, work {self.work}")
        if os.path.exists(self.flag("DONE")) or os.path.exists(self.flag("GAVEUP")):
            log("flags/DONE or flags/GAVEUP exists: nothing to do")
            return 0
        if not self.check_progress():
            return 3
        stages = ["s0", "s1", "s2", "s3", "s4"]
        for st in stages:
            log(f"=== stage {st}")
            getattr(self, st)()
            if a.until == st:
                log(f"--until {st}: stopping here")
                return 0
        open(self.flag("DONE"), "w").write(time.strftime("%F %T\n"))
        log("DONE")
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/flowline_rect_coarse.toml")
    ap.add_argument("--until", choices=["s0", "s1", "s2", "s3", "s4"])
    ap.add_argument("--dt", type=float, default=10.0, help="S1 time step (yr)")
    ap.add_argument("--steps", type=int, default=100, help="S1 steps per chunk")
    ap.add_argument("--batch", type=int, default=5, help="S1 chunks per ess run")
    ap.add_argument("--rms-tol", type=float, default=1e-3)
    ap.add_argument("--gl-tol-km", type=float, default=0.5, help="S1 stops only if the grounding line moved less than this in the last batch")
    ap.add_argument("--max-chunks", type=int, default=200)
    ap.add_argument("--heat-dt", type=float, default=1e4)
    ap.add_argument("--heat-steps", type=int, default=30)
    ap.add_argument("--map-steps", type=int, default=10, help="Newton's map: this many steps of [map] dt (500 yr)")
    ap.add_argument("--step-max-km", type=float, default=50.0)
    ap.add_argument("--step-min-km", type=float, default=1.0)
    ap.add_argument("--out-root", default="sweep_corrected")
    ap.add_argument("--poll-min", type=float, default=30.0)
    ap.add_argument("--max-spawns", type=int, default=3)
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()
    sys.exit(Pipeline(a).main())


if __name__ == "__main__":
    main()
