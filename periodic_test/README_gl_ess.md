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
* Stokes and free surface only (`--stokes-only`), from a 2D marine state (`ess transfer`, default
  `work/marine_flowline/run_cold_p268.15.npz`); three phases of 5 chunks of 100 yr (dt 5 yr): spin-up,
  retreat (`fcoeff` x 0.3), advance (`fcoeff` back, `aDIM` x 3). `--no-remask`: only Elmer moves the grounding line. `--bed temperate`: the bed at T0, so sliding
  is plain Weertman with `fcoeff` (on the cold start state's bed nothing moved in 1500 yr).
* `[elmer] periodic_gl` / `ess run --periodic-gl`: `elmerstreams` (ElmerStreams' contact.f90 on the raw loads,
  `Fw2` = Fw doubled on the master side), `fix` (SlidCoef_ContactPeriodic + the grounded mask made periodic after
  GroundedSolver), `none` (Elmer/Ice's SlidCoef_Contact).

Run (login node, repository root):

    git pull && bash periodic_test/run_gl_ess.sh               # prep job, then elmerstreams and fix in parallel
    CHUNKS=1 bash periodic_test/run_gl_ess.sh                  # a short trial first
    sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd analyse   # the tables again (each run job prints its own)

The table: after every chunk, the grounding line per row across y (y = 0 and y = W are the same physical row) and
the largest master (y = 0) / image (y = W) mismatch of the grounded mask, Zs, Zb, the velocities and the raw
vertical Stokes load. With `fix` everything but the raw load should match exactly; with `elmerstreams` the mask
(and through it Zb and the velocities) may not.

All these options are off by default; the template renders byte-identically without them (cached runs stay valid).
