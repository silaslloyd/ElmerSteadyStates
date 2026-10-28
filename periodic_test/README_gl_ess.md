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

## Thermomechanical version

Heat on (the cold/temperate mask moved between chunks, the y = W copies following y = 0; the grounding line by
contact only), iterative solvers throughout (Stokes as ElmerStreams: P1 + bubbles, GCR + ILU8, tolerance 1e-8, no
abort; heat BiCGStabL + ILU; free surfaces BiCGStab + ILU1). Phases: spin-up until the rms change per chunk < 2e-3
(at most 100 chunks = 10 kyr, every 5th chunk saved), retreat (accumulation x 0.3, 10 chunks), advance (x 3, 10).
The Stokes-only runs never reached a steady state before the forcing, and MUMPS ran out of workspace at 50 km.

    NOPREP=1 bash periodic_test/run_gl_ess.sh                                            # 20 km (mesh exists)
    GL_CONFIG=configs/test_periodic_gl_50km.toml NTASKS=256 NOPREP=1 bash periodic_test/run_gl_ess.sh   # 50 km

## Loads test: friction heating from the loads vs as a flux (current)

Both test configs: `[elmer] thermal = "loads"`, `sliding = "temperate"`. No cold/temperate mask (all cold and frozen,
`--bed cold`, the default of `run_gl_ess.py run`): the grounded bed temperature is free, no cap on T; the shelf base
at T0. Sliding as at the melting point everywhere (C = fcoeff (1 + 0.8 sin(2 pi y / W)), grounded), so the
dynamics are those of the temperate-bed scan: `--fbase 10` (the default now). Two runs, the same but for the
friction heating (`--heat`, `ess run --friction-heat`):

* `fix_loads`: frictionloads(loads, u) as nodal heat loads, from the periodic copy of the Stokes loads
  (`FixPeriodicLoads`, folded; run by the heat solver first as its Pre Solver, so this step's loads);
* `fix_flux`: C |u_b|^2 as a heat flux (per area, integrated by Elmer; no loads involved): the reference.
* `elmerstreams_loads`: none of the fix, ElmerStreams as it is: the raw loads halved at y < 1 in frictionloads,
  ElmerStreams' contact with Fw doubled on the master side (Fw2).

The same physics, so their bed temperatures should differ smoothly; a difference on the seam row larger than on the
middle row (y = W/2, where the bump is zero as on the seam) means the loads are wrong on the seam. Each run's table
adds `Tb seam` / `Tb mid` (the bed temperature's second difference across y on the seam / middle row: a seam stripe)
and `ploads z` (master/image mismatch of the periodic copy: 0). Other runs: `RUNS="MODE_HEAT ..."`. The periodic copy and a `Friction Loads` export from it are
made in every run (periodic_gl = fix).

    git pull && NOPREP=1 bash periodic_test/run_gl_ess.sh            # the three runs (20 km)
    sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd compare fix_loads fix_flux
    sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd compare elmerstreams_loads fix_flux

## ElmerStreams thermal chain (not used now)

`[elmer] thermal = "elmerstreams"` (both test configs) puts ElmerStreams Marine_Term's thermal treatment into the
ESS marine template (applied to the rendered input file; other configs unchanged):

* the cold/temperate mask set every time step by ElmerStreams' `ColdTempMaskSolver` (elmer/ColdTempMaskSolver.F90,
  copied unchanged): cold -> temperate when T - Tm >= 1e-3, temperate -> cold when the water sheet is gone;
* the water sheet `SheetSolverhw` (elmer/SheetSolverhw.F90, copied from SRC/sheetsolverhw.f90) with Marine_Term's
  hydrology material functions and parameters (elmer/ess_hydro.lua, copied from utils.lua / parameters.lua), fed by
  the melt from the heat solver's loads (`getmeltrate`), effective pressure from `Stress 3` (`ComputeDevStress`);
* friction heating from the Stokes nodal loads (`frictionloads`) and strain heating;
* the mask and the water sheet are carried from chunk to chunk (frozen fields updated from Elmer's end values).

The treatments differ in the loads:
* `elmerstreams`: raw loads, friction heating and melt halved on the y < 1 side (Coordinate 2 passed as y);
* `fix`: periodic copies (`FixPeriodicLoads`): the Stokes loads run by the heat solver first (`Pre Solvers`), the
  temperature loads (folded) and the strain-heating boundary weights (split) before the sheet solver; no halving;
* `none`: raw loads, nothing halved.

Needs a new prep (the water sheet is a new frozen field): `bash periodic_test/run_gl_ess.sh` (no NOPREP).
