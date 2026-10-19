# Periodic-BC / partitioning test

The ElmerStreams 3D stream runs show unconverged-looking temperatures along partition boundaries when
they run with periodic side boundaries on several partitions. This is a small case that separates the
possible causes.

## The case

A slab 30 km x 8 km x 600 m on a flat bed:
- **Mesh:** a 60 x 16 footprint, extruded by ElmerSolver into 10 layers, like the ElmerStreams meshes.
- **Flow:** driven along x by gravity tilted by a 0.005 slope; walls at x = 0 and x = L.
- **Physics:** steady Stokes (Newtonian, linear sliding), then steady heat with advection.
- **Thermal boundaries:** geothermal flux plus frictional heating at the bed, 243.15 K at the surface.

**Nothing depends on y**, so the exact discrete solution is uniform across the flow. Any structure in
y is an error, at partition interfaces or anywhere else. The sides y = 0 and y = W are either periodic
or free-slip walls, and the exact solution is the same for both.

Every case is compared with the serial run of the same physics, which uses direct solvers.

## Hypotheses and the cases that test them

| group | changes | tests |
|---|---|---|
| R | serial, direct | the boundary conditions and heating alone: are periodic + loads y-uniform at all? (`*_halve`: the `y < 1` halving in `frictionloads`) |
| A | partitionings, flux heating, direct | periodic BCs + partitioning alone. `xcut` (N x 1) never splits a periodic pair; `ycut` (N/4 x 4, as `stream.cmd`) splits them; `_per` adds ElmerGrid `-periodic 0 1 0` (as `stream.cmd`) |
| B | nodal friction loads in parallel, direct | loads at shared nodes / periodic nodes (`Flow Solution Loads` x velocity as `Temperature Load`) |
| C | heat solver GCR+ILU2 (ElmerStreams), BiCGStabL+ILU1, GCR without preconditioner | the preconditioner with periodic constraints across partitions |
| D | Stokes GCR+ILU8 with `Abort Not Converged = False` (ElmerStreams `linsys/gcr.sif`) | an unconverged Stokes solve feeding bad loads/velocities to the heat solve |
| E | Stokes P1/P1 stabilised instead of P2/P1 | periodic projectors with the P2 edge dofs |
| F | the full ElmerStreams combination, and the same with flux heating | does the test reproduce the problem, and does flux heating cure it? |

`python3 periodic_test/run_tests.py --list` prints every case.

## Running it (tethys, from the repository root)

```bash
git pull
sbatch periodic_test/run.cmd                 # all cases: ~30 small runs on 8 tasks
sbatch periodic_test/run.cmd --only R_ A_    # some groups (their serial references are added)
python3 periodic_test/run_tests.py --analyse # the table again, after the runs
```

- **Where the runs go:** `periodic_test/out/np<N>/<case>/`, with `case.sif`, `params.lua`, `elmer.log` and
  the per-partition outputs `out.bin.<pe>`.
- **Partitioned meshes:** `periodic_test/out/np<N>/meshes/<partitioning>/`.
- **Requirements:** `elmer/SteadyStateIO.so`, which `run.cmd` builds if it is missing, and ElmerGrid
  on the PATH.
- **Number of partitions:** N = the job's task count (`--ntasks`, default 8; a multiple of 4).
  `ycut*` is `-partition N/4 4 1` as in `stream.cmd`, `metis*` is `-metiskway N`, `xcut` is
  `-partition N 1 1` (skipped above N = 60, the footprint's columns). Results go to `out/np<N>/`, so
  runs with different N are kept side by side:
  `sbatch --ntasks=32 periodic_test/run.cmd`.

## Reading the table

Units are K, and relative velocity for `du_rel`.

- **`T_y_spread`, `Tbed_y_spread`, `u_y_spread`:** the largest range across y at fixed (x, z). The
  exact value is 0; for a correct set-up it is at round-off or solver-tolerance level.
- **`dT_max`:** the largest |T − T_ref|, against the serial direct reference of the same physics.
  - `dT_iface`, `dT_inter` and `dT_side` split it over partition-interface nodes, interior nodes and
    the periodic sides. A problem located at partition boundaries shows as `dT_iface` ≫ `dT_inter`.
- **`mismatch`:** the largest disagreement between partitions about the value at a shared node. It
  should be 0.
- **`worst_at`** (in `summary.csv`): where the largest deviation from the across-flow mean sits, and
  whether that node is on an interface or a periodic side.
- **`status`:** whether Elmer finished, whether any linear solves did not converge, and whether the
  extruded bed/surface conditions (5, 6) were applied.

If plotly is installed, `out/np<N>/<case>/bed.html` maps the basal temperature minus its across-flow mean,
with the partition-interface nodes ringed.

## Transient cases (T) and ParaView

Some artefacts only appear after many time steps, so the T cases run 200 steps of 50 yr with
temperature-dependent sliding: C·exp(−min(T − T0, 0)/1 K). Each step solves Stokes, then heat, so
sliding and temperature feed back on each other and a small partition error can grow. They start from a
linear profile, 270 K at the bed and 255.15 K at the surface, so the bed warms towards melting during
the run.
- **Variants:** flux or nodal-load heating (loads halved at y = 0, as in ElmerStreams), each in serial,
  y-cut with `-periodic`, and with your iterative solvers on the y-cut and METIS partitionings.
- **Output:** every case writes VTUs every 10 steps (`out_every`) to `out/np<N>/<case>/vtu/`. Open
  `case_t*.pvtu` in ParaView, as one time series, to watch an artefact develop. Apply the
  *Process Id Scalars* filter to colour the partitions.
- **Run them:** `sbatch periodic_test/run.cmd --only T_`

## Looking at the basal temperature on the footprint

The analysis also writes `out/np<N>/<case>/bed.npz`: basal temperature, its difference from the
reference, the nodal friction load, partition ownership and the interface nodes, for every bed node.
These files are small. Copy them to your laptop and plot them there:

```bash
rsync -av --include='*/' --include='bed.npz' --include='summary.csv' --exclude='*' \
    <tethys>:<repo>/periodic_test/out/ periodic_test/out/
python periodic_test/plot_bed.py periodic_test/out/np8      # or set DIR and press Run
```

`bed_maps.html` has a menu to pick the case. Its rows (x along the flow, y across):
1. basal temperature;
2. basal temperature minus its across-flow mean, where any artefact shows (exactly 0 without one);
3. basal temperature minus the serial reference;
4. the friction load minus its across-flow mean (loads cases);
5. the partitions.

Black dots mark the partition-interface nodes, so you can see whether an artefact follows them.
To make the files again from finished runs: `python3 periodic_test/run_tests.py --analyse --np N`.

## Things not covered yet

- **Mortar periodic BCs** (`Mortar BC`, `Apply Mortar BCs`), Elmer's parallel-aware alternative to
  conforming periodic projectors.
- **The temperature limiter, cold/temperate masks and transient time stepping.** The test is linear
  and steady on purpose. Add them once one of the groups above reproduces the problem.
