# Periodic grounding-line test on the ElmerStreams setup

`run_gl_ess.py`, `run_gl_ess.sh`, `configs/test_periodic_gl.toml`. The ESS marine template is ElmerStreams'
Marine_Term setup (bedrock, filler, ice, bed interface and surface bodies; contact grounding line moving every
step; Buoyancy sea pressure and spring). The test makes it periodic across the flow and moves the grounding line
back and forth:

* `[case] width = 20 km`, `y_elements = 4`, `periodic_sides = true`: side 2 the periodic image of side 1 for the
  bedrock (BC 8 -> 7), filler (12 -> 11) and ice (16 -> 15), as in ElmerStreams' commented periodic lines; no
  `Velocity 2 = 0` on the ice sides; `ElmerGrid -periodic 0 1 0` (partitioned `-partition N 1 1`).
* `[case] slide_ybump = 0.8`: the sliding coefficient times 1 + 0.8 sin(2 pi y / width) (0.2 .. 1.8), so the
  grounding line is not straight and there is shear across the seam (0.3 left it straight).
* The footprint is refined to 500 m within +- 20 km of the start state's grounding line (~739 km;
  `case/flowline_marine_gl.geo.in`, `gl_from_state`), so the grounding line can differ between rows.
* Stokes and free surface only (`--stokes-only`), from the 2D cold marine state (`ess transfer`,
  `work/marine_flowline/run_cold_p268.15.npz`, grounding line ~739 km) with the whole bed temperate
  (`--bed temperate`) and the sliding coefficient `fcoeff` x `--fbase`. `--no-remask`: only Elmer moves the
  grounding line. On the cold bed nothing moved in 1500 yr; with fbase 1 the grounding line ran to the calving
  front within 300 yr. So first a scan of fbase (spin-ups only, `SCAN="10 30 100"`), then the full test with the
  fbase that keeps the grounding line inside (10: all of 10, 30, 100 advanced, 10 the wavier): spin-up (5 chunks
  of 100 yr, dt 5 yr), retreat (10: friction x 30, accumulation x 0.3), advance (10: back to the base friction).
* `[elmer] periodic_gl` / `ess run --periodic-gl`: `elmerstreams` (ElmerStreams' contact.f90 on the raw loads,
  `Fw2` = Fw doubled on the master side), `fix` (SlidCoef_ContactPeriodic + the grounded mask made periodic after
  GroundedSolver), `none` (Elmer/Ice's SlidCoef_Contact).

Run (login node, repository root):

    git pull && SCAN="10 30 100" bash periodic_test/run_gl_ess.sh   # spin-ups with fcoeff x 10, 30, 100 (fix)
    FBASE=10 bash periodic_test/run_gl_ess.sh                  # then the full test (elmerstreams and fix)
    CHUNKS=1 FBASE=10 bash periodic_test/run_gl_ess.sh         # a short trial
    sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd analyse   # the tables again (each run job prints its own)

The table: after every chunk, the grounding line per row across y (y = 0 and y = W are the same physical row) and
the largest master (y = 0) / image (y = W) mismatch of the grounded mask, Zs, Zb, the velocities and the raw
vertical Stokes load. With `fix` everything but the raw load should match exactly; with `elmerstreams` the mask
(and through it Zb and the velocities) may not.

All these options are off by default; the template renders byte-identically without them (cached runs stay valid).
