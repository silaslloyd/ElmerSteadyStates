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
| A | partitionings, flux heating, direct | periodic BCs + partitioning alone. `x8` never splits a periodic pair; `y4x2` splits them; `_per` adds ElmerGrid `-periodic 0 1 0` (as `stream.cmd`) |
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

- **Where the runs go:** `periodic_test/out/<case>/`, with `case.sif`, `params.lua`, `elmer.log` and
  the per-partition outputs `out.bin.<pe>`.
- **Partitioned meshes:** `periodic_test/out/meshes/<partitioning>/`.
- **Requirements:** `elmer/SteadyStateIO.so`, which `run.cmd` builds if it is missing, and ElmerGrid
  on the PATH.
- **Partitioning keywords:** `stream.cmd` uses `-metiskway`; `y4x2*` is `-partition 2 4 1`, its
  `$((N/4)) 4 1` with N = 8.

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

If plotly is installed, `out/<case>/bed.html` maps the basal temperature minus its across-flow mean,
with the partition-interface nodes ringed.

## Things not covered yet

- **Mortar periodic BCs** (`Mortar BC`, `Apply Mortar BCs`), Elmer's parallel-aware alternative to
  conforming periodic projectors.
- **The temperature limiter, cold/temperate masks and transient time stepping.** The test is linear
  and steady on purpose. Add them once one of the groups above reproduces the problem.
