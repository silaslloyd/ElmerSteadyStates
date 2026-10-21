# ElmerSteadyStates

Steady states (stable *and* unstable), pseudo-arclength continuation and linear stability for Elmer/Ice
models. It drives an unmodified Elmer time stepper from Python.

The first case is a thermo-coupled, land-terminating flowline with subtemperate sliding, a free
surface, and a bedrock and filler layer. It is the Land_Term set-up of
[ElmerStreams](https://github.com/silaslloyd/ElmerStreams), rebuilt here so that this repository is
self-contained: `ess setup` makes its own mesh and initial conditions. The parameter followed here is
the surface temperature `Tsurf`.

## Idea

Elmer only time-steps, and a time stepper can never converge to an unstable steady state. So instead
of the time-stepper's own iteration, the driver solves for the steady state directly:

* **State.** `x = (Temperature on all nodes of bedrock, filler and ice, Zs on the surface nodes)`.
  The velocity is slaved to `x` and recomputed by Stokes at every step, so it is not part of the
  state.
* **Map.** `Phi(x)` = one Elmer run of `n_steps` of size `dt` starting from `x`. The state is read
  and written exactly by `elmer/SteadyStateIO.F90`: binary doubles keyed by global node number, so
  there is no restart file and no ASCII rounding.
* **Steady states.** These are fixed points, `F(x) = Phi(x) - x = 0`. They are solved by
  **Newton–Krylov** (Tuckerman's method):
  * GMRES on `J = DPhi - I`, with each `J v` computed as a finite difference, i.e. one Elmer run.
  * The time stepper itself is the preconditioner. Modes that decay fast within the horizon
    `T = n_steps dt` have `DPhi ≈ 0`, so `J ≈ -I` on them. GMRES therefore only has to resolve the
    slow modes and the unstable ones.
  * Newton converges to unstable states as readily as to stable ones.
* **Continuation.** Pseudo-arclength continuation in one parameter `p` (any top-level assignment in
  `parameters.lua`):
  * The bordered Newton system is also solved matrix-free.
  * One Elmer run gives `J dy + F_p dp` at once.
  * Folds are passed and located: the parameter turns, and a quadratic estimate of `p_fold` is
    printed.
* **Stability.** Arnoldi on `DPhi` at a steady state, at one Elmer run per Krylov vector.
  * Eigenvalues `mu = exp(sigma T)` give `sigma = log(mu)/T`.
  * Because the steps are backward Euler, `sigma_be = (1 - mu^(-1/n))/dt` is also reported.
  * `|mu| > 1` means unstable.

Steady states of the staggered scheme (Stokes then heat then free surface in each step) are exactly
the steady states of the coupled equations. They do not depend on `dt` or on the coupling
iterations; only the transient path does.

## The case

* `case/parameters.lua`: the physical parameters, taken from ElmerStreams Land_Term with the
  hydrology parameters removed.
* `case/flowline.geo.in`: the footprint geometry. It is one element wide, refined around the margin,
  and has the same boundary numbering as ElmerStreams.

`ess setup` writes everything else into `<work>/case/`:

* **Mesh.** Built with gmsh and ElmerGrid, and partitioned for the number of MPI tasks.
* **Initial conditions.**
  * Geometry: a steady shallow-ice profile for linear viscous ice without sliding, on the linear bed,
    for the model's own accumulation. The margin falls at 0.8 L.
  * Temperature: conduction only.
  * Bed: cold everywhere under the ice.
  * Beyond the margin: a 30 m "unglaciated" layer (GroundedMask = 0), as in ElmerStreams.

This is just a first guess for Newton. Elmer's built-in `FreeSurfaceSolver` is used; ElmerStreams
used a copy of it.

### Two margin set-ups

* **`configs/land_flowline.toml`: full domain with a regularised, free margin.** The domain runs to
  L. Beyond the margin, ablation fades out smoothly as the ice thins towards 30 m (`ess_Hs`,
  `ess_Hmin` in `elmer/ess_utils.lua`, or overridden from a `[lua]` config section). The edge is a
  steep front within about one element. Its shape is poorly determined, which makes Newton and
  continuation struggle there.
* **`configs/land_margin.toml`: fixed margin (recommended).** The mass balance is fixed in x, so the
  steady flux is the mass balance integrated from the divide. The margin therefore sits where that
  integral vanishes, at x = 0.8 L, whatever Tsurf, sliding or temperature. The domain ends there
  (`[case] x_end = 0.8`, geometry `case/flowline_cut.geo.in`), with:
  * no flow through the end;
  * Zs clamped to bed + `edge_thickness` (30 m) at the end node, by equal `Min Zs`/`Max Zs` bounds
    that are always active;
  * the plain mass balance, with no thickness factor.

  To start it from a converged regularised state, carry that state onto the new mesh with
  `ess transfer`:

  ```bash
  sbatch jobs/ess.cmd setup configs/land_margin.toml
  sbatch jobs/ess.cmd init  configs/land_margin.toml
  python3 -m ess transfer configs/land_margin.toml --from work/land_flowline/steady_p268.15_soft.npz
  sbatch jobs/ess.cmd newton configs/land_margin.toml --state work/land_margin/start_p268.15.npz
  sbatch jobs/ess.cmd continue configs/land_margin.toml --state work/land_margin/steady_p268.15.npz
  ```

  `transfer` interpolates column by column: temperature on the normalised height in the ice and on
  the depth in the rock, and Zs in x. It takes the cold/temperate mask from the source.

The frontal sliding zone (C/2 within 10 km of x = L) lies beyond the ice in both set-ups. Set
`[lua] ess_front` (in m) to move it under the ice.

### Marine flowline

`configs/marine_flowline.toml` (`[case] geometry = "marine"`) is the ElmerStreams `Marine_Term`
set-up:
* **Domain:** an 816 km flowline on a bed from −96 m to −224 m below sea level, deepening seaward.
* **Ice:** grounded to about 686 km, then an unconfined shelf to a fixed calving front at L.
* **Mass balance:** uniform accumulation `aDIM`, no basal melt.
* **Initial state:** read from Marine_Term's `Initialisation/Dimensional_ICs` tables, copied to
  `case/marine_ics`.
* **Mesh:** refined around the grounding line (`case/flowline_marine.geo.in`).

`elmer/steady_template_marine.sif` differs from the land template in these ways:
* **Base:** the base Zb is a free surface (its own solver) and part of the state (`[state]
  variables = ["Temperature", "Zs", "Zb"]`). There is no Zb limiter.
* **Sea pressure:** on the floating base and the calving front.
* **Sea spring:** `elmer/Buoyancy.F90`, the ElmerStreams copy, built by `make -C elmer`. It vanishes
  at a steady state, so it does not move the fixed points.
* **Frozen grounded mask,** like the cold/temperate mask:
  * grounded (1): Zb on the bed, no normal velocity, Weertman sliding;
  * floating (−1): no friction, T = T0 at the base.

  Within a run the grounding line cannot move, so the map is smooth.
* **Moving the grounding line:** `[masks] auto` (on for marine) checks the masks after every
  converged Newton solve and every branch point, using flotation, H vs H_f = ρw/ρi·(−bed):
  * floating nodes thicker than flotation are grounded;
  * grounded nodes thinner than flotation are floated, with their base put at flotation;
  * the cold/temperate mask is updated too.

  Newton is then re-run, until nothing changes. The grounding line therefore moves node by node
  along a branch (column `remasked` in `branch.csv`).

  ```bash
  sbatch jobs/ess.cmd setup    configs/marine_flowline.toml
  sbatch jobs/ess.cmd init     configs/marine_flowline.toml
  sbatch jobs/ess.cmd run      configs/marine_flowline.toml --chunks 15 --rock-cap 0.001
  sbatch jobs/ess.cmd newton   configs/marine_flowline.toml --state work/marine_flowline/run_p268.15.npz
  sbatch jobs/ess.cmd continue configs/marine_flowline.toml --name flux --state work/marine_flowline/steady_p268.15.npz
  ```

## Changes relative to `Land_Term/ThermoCoupled_template.sif`

`elmer/steady_template.sif` keeps the same mesh, extrusion, bodies, materials, Stokes, heat and free
surface solvers, and the same sliding law and boundary conditions, except for these changes:

* **No strain heating.** The deformational-heat solver and the ice `Heat Source` are removed.
  Frictional heating at the bed (`Temperature Load` from Stokes loads × velocity) is kept.
* **No hydrology.** The sheet solver, effective pressure and melt source are removed; sliding never
  used them.
* **Frozen masks.** `ColdTempMask`, `GroundedMask`, `Zb` and `bedrock` are taken once from the first
  state and then read from the state file in every run; they are never recomputed (see `remask`). Mask switching is
  non-smooth, and Newton needs a smooth map. The driver reports, at every converged state, where the
  frozen mask is inconsistent:
  * `cold_above_Tm`: cold-bed nodes above the melting point.
  * `temp_freezing`: temperate nodes where the bed refreezes.
* **Autonomous.** Neither accumulation nor surface temperature depends on time.
* **No noise field and no GL-distance solver.** The 10 km frontal zone uses `L - x` instead.
* **Diagnostics removed:** the stress, Fw, distance and grounded/cold mask solvers. The limiter is
  off.
* **Direct solvers.** MUMPS is used for Stokes, heat and free surface, with
  `Abort Not Converged = True`. This makes the map deterministic and accurate enough for
  finite-difference Jacobians.

Things the driver does **not** change, worth knowing when comparing with the analytic S-curve:

* The surface condition is a Robin (heat transfer coefficient 1 in MPa m yr⁻¹ K⁻¹, i.e. about a
  66 m conductive skin), not a Dirichlet condition. Its external temperature is the constant
  `Tsurf` everywhere (ElmerStreams ramped it to T0 under ice thinner than 100 m).
* The bed temperature is not capped at T0 on the cold part of the bed.
* `frictionloads` halves the frictional heat at nodes with y < 1 (kept from ElmerStreams). On this
  one-element-wide flowline that makes the y = 0 side get half the heating of the y = W side. It is
  probably worth removing (`elmer/ess_utils.lua`).

## Setup (cluster)

```bash
git clone https://github.com/silaslloyd/ElmerSteadyStates.git
cd ElmerSteadyStates
module load elmerfem
make -C elmer                      # builds elmer/SteadyStateIO.so with elmerf90
```

Needs ElmerSolver, ElmerGrid and gmsh (as for ElmerStreams), and Python 3 with numpy (Python ≥ 3.11,
or `pip install tomli`). The driver runs serially inside the batch job and launches
`mpirun -np $SLURM_NTASKS ElmerSolver` for every map evaluation.

## Workflow

Run each command with `sbatch jobs/ess.cmd <command> configs/land_flowline.toml [options]`. Use the
same `--ntasks` for every job, because the mesh is partitioned for it in `setup`; a flowline doesn't
need many, and 32 to 64 is plenty. Everything is written under `work/land_flowline/`, and logs go to
`work/land_flowline/ess.log`.

1. **`setup`**: builds the mesh, the initial-condition tables and `ics.lua`.
2. **`init`**: one Elmer run from those initial conditions. Writes `state0.npz`, containing the
   state, the frozen fields and the node coordinates. To start from an existing Elmer `.result`
   instead, use `extract --restart FILE`.
3. **`check`**: verifies the map is fit for Newton.
   * `roundtrip` must say *exact*.
   * `repeat` (two identical runs) should be far below `[fd] eps`.
   * `linearity` prints `J v` for decreasing perturbation sizes. Pick `[fd] eps` inside the plateau.
   * `residual` shows how far the state is from steady.
4. **`run --chunks N`**: plain time integration. Use it to spin the first guess up, or to find the
   largest `dt` for which the free surface stays stable. It writes `run_p<P>.npz`.
5. **`newton [--state F] [--p P]`**: steady state at fixed `p` → `steady_p<P>.npz`.
6. **`continue --state F`**: the branch, written to `branch_<name>/`:
   * `branch.csv` holds one row per point: parameter, arclength, residual, iteration counts, fold
     flag and mask diagnostics.
   * `points/state_XXXX.npz` holds each full state.
   * Set `stability_every = k` in `[continuation]` for a short Arnoldi at every k-th point, which
     marks the points as stable or unstable.
7. **`stability --name <branch> --point K`** (or `--state file.npz`): leading eigenvalues and
   eigenvectors → `stability/<label>/stability.npz`.
8. **`remask --state F`**: re-freezes the cold/temperate mask from a state (marine: the grounded mask
   too). Run it when a state reports `cold_above_Tm > 0` or `temp_freezing > 0` (marine: `to_grounded`,
   `to_floating`), then continue from the new file. `[masks] auto = true` does this inside `newton`
   and `continue`.

9. **`spectrum --state F [--p P]`**: the steady states at one parameter value with temperate beds of
   increasing size (`ess/spectrum.py`). For k = 0, 1, 2, ... it does the following:
   * **Mask:** temperate (T = T0) on the k glaciated bed columns nearest the terminus, cold everywhere
     else.
   * **Newton:** at the fixed parameter, from the previous k's state, with no remasking. The bed may go
     above melting during the solve.
   * **Thermal check:** the state is discarded if any cold bed node ends above melting (`--tol-T`).
   * **Hydrology check:** the steady hydrology on the patch is solved (`hydrology_steady`, settings in
     `[hydrology]`):
     * the water flux q = ∫ m dx, from the cold/temperate boundary (q = 0) to the margin, with the melt
       rate m from the temperature loads;
     * then the sheet h from the margin (h = h0) back upstream through q = K0 h^α d/dx(c_N/h − φ0), with
       ElmerStreams' φ0 = −σ_zz + ρ_w g z_b.

     The state is discarded if h reaches zero inside the patch.

   Kept states go to `<work>/spectrum_p<P>/points/state_XXXX.npz`, with XXXX = k. Every k gets a row in
   `spectrum.csv`, with the checks and the reason for any discard. `plot_branch_states.py` with that
   folder shows them against the boundary position. `--k-max` and `--k-step` limit the work, and
   resubmitting resumes.

A job that times out can simply be resubmitted with the same command:

* Every finished Elmer run is cached by a hash of its inputs, so no run is repeated.
* Continuation resumes from its last two points.
* Arnoldi resumes from its checkpoint.

**Plots**: rsync `work/` back to your machine, without `cache/` and `runs/`, then run in the IDE
(plotly, linear axes):

* `plots/plot_branch.py`: bifurcation diagram, with folds and stability marked.
* `plots/plot_state.py`: basal temperature, surface and temperature field of a state, or of an
  eigenmode from a `stability.npz`.

## Choosing the maps

* **`[map]`, for Newton and continuation.** Use the largest `dt` the free surface tolerates, with a
  horizon `T` long enough to damp the fast modes.

  `T` must not be *too* long on unstable branches. A mode growing at rate `sigma` is amplified by
  `exp(sigma T)` per evaluation, and once `sigma T` is more than a few the finite-difference
  products leave the linear range. Continuation then has to take tiny steps, or Elmer fails (the toy
  model shows this). If that happens, shorten `T`.
* **`[stability_map]`.** Needs `|sigma| dt << 1` for the modes you care about. Backward Euler damps
  a mode with `sigma dt > 2` to `|mu| < 1`, so such an unstable mode would be reported as stable.

  The staggered coupling also has an O(dt) error in growth rates (2–5 % at `sigma dt ≈ 0.03` in the
  toy). Its sign is robust; check its magnitude by halving `dt`.
* **`[parameter] scale`.** The parameter change that counts as much as one scaled unit of state
  change in the arclength.
  * Too small, and the steps crawl.
  * Too large, and the branch looks sharply curved at the fold, so steps there become tiny.

## Stability to perturbations across the flow (3D strip)

The steady states are uniform across the flow (y), so their linear modes are
cos(n pi y / W) x (shape in x and z): each spanwise wavenumber k_y = n pi / W has its own modes. Along
the flow (x) the steady state varies, so there are no k_x modes; each k_y has a set of global modes in
x. (A local (k_x, k_y) dispersion relation needs a local slab analysis at a fixed x.)

`configs/land_margin_3d.toml` runs the same model on a strip `width` wide with `y_elements` elements
across and free-slip side walls (`[case] width`, `y_elements`; `case/flowline_cut.geo.in`):

```bash
sbatch jobs/ess.cmd setup configs/land_margin_3d.toml
sbatch jobs/ess.cmd init  configs/land_margin_3d.toml
python3 -m ess transfer configs/land_margin_3d.toml --from work/land_margin/steady_p268.228_flux.npz
sbatch jobs/ess.cmd newton configs/land_margin_3d.toml --state work/land_margin_3d/start_p268.228.npz
sbatch jobs/ess.cmd stability configs/land_margin_3d.toml --state work/land_margin_3d/steady_p268.228.npz --spanwise
```

* `transfer` copies a 2D steady state onto the strip, uniform in y. Newton then removes the small
  differences in discretisation; by symmetry the state stays uniform in y.
* `stability --spanwise` (ess/spanwise.py) removes the across-flow mean from every Arnoldi vector, so
  the 2D modes (n = 0) drop out and Arnoldi finds the least stable modes with n >= 1. Each mode is
  labelled by its dominant n (a discrete cosine transform across the flow), with k_y and the
  wavelength 2 W / n, in the log and in `stability.npz` (`spanwise_n`, `spanwise_wavelength`,
  `spanwise_spectrum`).
* `plots/plot_mode_3d.py`: x-y maps of a mode's basal temperature and surface perturbations, the
  spectrum of every mode, and growth rate against wavelength.
* A strip of width W resolves wavelengths 2W, W, 2W/3, ... down to about 2W/(y_elements/2). For a
  dispersion curve, run several widths (copy the config, change `width` and `work_dir`).
* Cost: each run is about y_elements times a flowline run. MUMPS (the default) slows down sharply in
  3D; `[elmer] linear_stokes = "iterative"`, `linear_heat = "iterative"` switch to GCR / BiCGStabL with
  ILU (`stokes_preconditioner`, `iterative_tol`). Their error is noise in the Jacobian products, so
  check with `ess check --vector residual` (the repeat difference should stay far below `[fd] eps`).
  If GCR + ILU does not converge on the P2/P1 Stokes system, `stokes_element = "Stabilized"` (P1/P1)
  is much easier for ILU, but changes the discretisation.

## Non-smooth parts (where Newton can struggle)

* **The frozen masks.** Their consistency is checked at every converged state. If a branch runs into
  `cold_above_Tm > 0`, the state is no longer physical with that mask. Re-freeze the mask from the
  new state (re-extract) and continue piecewise.
* **The margin** (regularised set-up only). The free-surface `Min Zs` limiter and the steep, poorly
  determined edge. The fixed-margin set-up avoids both.
* **The `max(·, 0)` in the friction loads.**
* **The `T > Tm` cap inside the sliding law.**

A Jacobian across any of these kinks is poor. `check` (the linearity plateau) shows whether that is
happening at the current state.

## Testing without Elmer

`ess/toy.py` is a stand-in model: an ice column with subtemperate frictional heating plus a
thickness variable. Its S-curve and fold are known exactly. `python3 -m ess toy-demo` runs the whole
pipeline on it, from Newton through continuation round the fold to stability.

The tests in `tests/test_solvers.py` check:

* the unstable middle branch and its growth rate;
* Arnoldi against the exact map Jacobian;
* the fold location to 1e-5;
* resuming;
* the Elmer plumbing, end to end, with `tests/fake_elmer.py` in place of `mpirun ElmerSolver`:
  partitioned files, shared nodes, extract, check, newton and cache.

```bash
python3 -m pytest tests
```

`SteadyStateIO.F90` and the sif template have not been run through Elmer yet (no Elmer where this
was written). Expect the first `make` / `extract` / `check` to need small fixes; `check` is designed
to catch them.

## Layout

```
case/           parameters.lua, flowline.geo.in
ess/            driver: case (mesh + initial conditions), statefile (binary state files, layout),
                model (map, scaled problem), elmer (Elmer runs, cache, mask report),
                linalg (GMRES, Arnoldi), newton, continuation, stability, toy, config, cli
elmer/          steady_template.sif, ess_utils.lua, SteadyStateIO.F90, Makefile
configs/        land_flowline.toml
jobs/           ess.cmd (SLURM, mpirun)
plots/          plot_branch.py, plot_state.py
tests/          test_solvers.py, fake_elmer.py
```
