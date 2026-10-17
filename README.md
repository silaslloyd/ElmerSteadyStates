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
8. **`remask --state F`**: re-freezes the cold/temperate mask from a state. Run it when a state
   reports `cold_above_Tm > 0` or `temp_freezing > 0`, then continue from the new file.

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
