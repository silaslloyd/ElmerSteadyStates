# fssa3d: free-surface time-step tests in 3D

**Main route (ESS machinery, starting from the converged temperate flowline):**

1. `SRC_CFG=work/flowline_gl_fine/refined/pipe1.toml SRC_STATE=<converged flowline .npz> sbatch fssa3d/prep.cmd`
   writes `configs/flowline_gl_3d.toml` (the flowline's mesh along x, 20 km wide, free-slip walls, cells >= 500 m),
   builds the 3D mesh and extrudes the flowline state laterally (`ess transfer`) -> `work/flowline_gl_3d/start_flowline.npz`.
2. `bash fssa3d/sweep.sh` (DT, YEARS, DIP, NT, VARIANTS): one job each of `jobs/flowline_fssa.cmd`, Stokes-only, temperate
   sliding, friction dip `1 - DIP exp(-((y - W/2)/2 km)^2)` ([case] slide_ydip), for FSSA full /
   none (and adapted), explicit and coupled (implicit) free surfaces, plus dip-free controls.
3. `python3 fssa3d/ess_compare.py`: grounding line (centre vs wall), surface / base change from the start, cross-flow
   surface relief and grid-scale roughness per run, and each run against its control.

The start state is a steady state of the flowline's discretisation (P2/P1 from the pipeline); the 3D jobs use the
cheaper P1/P1 + bubbles + GCR by default, so the controls show how much the extruded state drifts by itself.

**Standalone route (stock Elmer + the repo's contact/buoyancy libraries, no ESS driver):** `make_case.py` (grounded slab,
bump relaxation), `make_marine.py` (small marine strip from a made-up start), `run.cmd` (sweep over dt/scheme), `analyse.py`.
