# Periodic-sidewall loads test

The grounding-line contact and the nodal friction heating use Elmer's nodal Stokes loads (`Flow Solution Loads`).
Along periodic boundaries they come out non-periodic even where the velocity and pressure are periodic; ElmerStreams
halves them on one side. This test measures exactly how they differ from the right values, node class by node class.

## The case

- **A parallel-sided slab** on a flat bed, 600 m thick, flowline footprint 100 km x W (2 km cells, W = 1, 2 or 4
  cells across), extruded into 10 layers. Gravity is tilted along x (slope 0.005), z is normal to the bed.
- **Periodic in x**, so nothing depends on x or y and the exact solution is known everywhere.
- **The sides y = 0 / y = W:** periodic (y = W the image of y = 0), or free-slip walls (`Velocity 2 = 0`) as the
  control. With one cell across (`ny1`) every node is on a periodic side, as in the flowlines.
- **Stokes only** (Newtonian, linear sliding with a normal-tangential bed), optionally with a free surface for
  20 x 0.5 yr (`fs_` cases). No heat.
- **Serial and 4 partitions** cut across x (`xcut`, `-periodic`), and METIS (`metis`).

## The exact loads

A node's exact load is the stress on the bed times the node's **own** share of the bed: a quarter of each bed face
next to it, not counting faces that only touch its periodic image. At the bed the stress is
rho g H (sin a, 0, cos a): the along-flow component is the basal friction, the vertical one the normal reaction.
Elsewhere (interior nodes and the free surface) the exact load is zero.

## The table (`python3 periodic_test/run_loads.py --analyse`)

Bed nodes are split into the master side (y = 0), the image side (y = W) and interior nodes; the x-periodic
columns are left out. For each:
- **`_z`, `_x`:** the measured load divided by the exact one (1 = right). `y0_z` = 2 and `yW_z` = 1 would mean
  the master carries its image's share as well (so halving the master is right); 1 and 1 that both are right.
- **`_y`:** the cross-flow load divided by the node's exact vertical load (exact 0): a force passed between a
  master node and its image.
- **`_w`:** Elmer's boundary weight divided by the node's own area (1 = the weights do not see the periodicity).
- **`_sum`** (partitioned cases): the same summed over the partitions' copies of shared nodes, and
  **`shared_copy_spread`**: how much the copies differ.
- **`top_*`:** loads at surface nodes relative to the bed's vertical load (exact 0).

## Running (tethys, repository root)

```bash
git pull
sbatch periodic_test/run_loads.cmd                    # every case (4 tasks, a few minutes)
python3 periodic_test/run_loads.py --analyse          # the table again
```
Results: `periodic_test/out_loads/<case>/` (case.sif, elmer.log, out.bin.<pe>) and `out_loads/summary.csv`.
