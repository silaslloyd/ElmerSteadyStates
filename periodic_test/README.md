# Periodic-BC / partitioning test

The ElmerStreams 3D stream runs show unconverged-looking temperatures along partition boundaries when they
run with periodic boundaries on several partitions. This test looks for the cause in the simplest setting
where the exact answer is known.

## The case

A parallel-sided slab on a flat bed:
- **Footprint:** square, 10 km × 10 km, 500 m cells by default (`--res` for finer), extruded by
  ElmerSolver into 10 layers, as the ElmerStreams meshes are. The slab is 600 m thick.
- **Boundaries:** periodic on all four sides, x = 0 ↔ x = L and y = 0 ↔ y = W, and ElmerGrid
  partitions with `-periodic 1 1 0`.
- **Flow:** z is normal to the bed, and gravity is tilted by a slope of 0.005, so a component of
  gravity drives the flow along x.
- **Physics:** Newtonian Stokes (P1/P1 stabilised) with linear sliding. The drag coefficient is
  either constant or temperature dependent, Cslip·exp(−min(T − T0, 0)/dT) as in ElmerStreams. Heat
  is solved with advection. There is no strain heating.
- **Thermal boundaries:** geothermal plus frictional heating at the bed, given either as a flux
  τ_b·u_b or as ElmerStreams' nodal loads; a fixed temperature at the surface.
- **Free surface (FS cases):** Zs is periodic too. StructuredMeshMapper moves the mesh before each
  step, and the ElmerStreams FreeSurfaceSolver, with its limiter, updates Zs after it. Accumulation
  is zero.

**Nothing depends on x or y, so the exact discrete solution is horizontally uniform.** That holds for
the temperature, the velocity and the surface. Any horizontal structure is an error, from the
partitioning, the periodic BCs or the linear solvers, at partition interfaces or anywhere else.

Nodal friction loads need halving on the master sides (x < 1 m, y < 1 m; both at the corner). The
master nodes carry the loads of their periodic images too. `R_loads` (no halving) against
`R_loads_halve` checks this.

## Cases

| group | what | tests |
|---|---|---|
| R | serial, direct, steady | the periodic BCs (including the doubly periodic corner) and the heating alone |
| A | partitionings, direct, steady, flux heating | partitioning with periodic BCs: boxes, strips across x, strips across y, METIS; `box_noper` = boxes without `-periodic` |
| B | nodal loads in parallel, direct | loads at shared and periodic nodes |
| C | ElmerStreams iterative solvers, steady | heat GCR+ILU2, Stokes GCR+ILU8 (Abort Not Converged = False): where the tolerance-level errors sit |
| T | transient, warm, sharp sliding (dT = 0.15 K), fixed surface | 100 × 100 yr from 272.5 K at the bed (surface 259.15 K): the sliding/heating feedback switches on and can amplify a seed |
| SS | steady, warm, coupled (up to 60 Picard iterations) | the same feedback without time stepping |
| FS | as T with a free surface | the free surface across partition and periodic boundaries |

`python3 periodic_test/run_tests.py --list` prints every case.

**Free-surface time step.** Without stabilisation the free surface is unstable above about 1 yr. That
is the relaxation time of the fastest surface mode for this slab (600 m, 10¹⁴ Pa s, sliding at
melting), about 0.6 yr. The FS cases therefore default to 0.5 yr × 2000 steps (1 kyr), with a snapshot
every 100 steps. The relevant options:
- `--fs-dt`, `--fs-steps`, `--fs-every` change these.
- `--nsvec <ElmerStreams>/SRC/IncompressibleNSVec.so` uses the FSSA-capable Stokes solver and turns
  FSSA on (θ = 1, "full"), which allows a larger `--fs-dt`.
- `--fs-lib <ElmerStreams>/SRC/FreeSurfaceSolver.so` uses your free-surface build; it is the module
  version with older allocation code.

## Running (tethys, repository root)

```bash
git pull
sbatch periodic_test/run.cmd                              # every case, 16 tasks, 500 m
sbatch periodic_test/run.cmd --only R_ A_                 # some groups (serial references added)
sbatch periodic_test/run.cmd --res 100 --box 4 4 --only FS_ --no-refs   # high resolution
python3 periodic_test/run_tests.py --np 16 --analyse      # the table again (same options as the run)
```

- **Where the results go:** `periodic_test/out_sq/np<N>[_dx<res>][_fsdt<dt>][_fssa]/<case>/`, with the
  meshes under `meshes/`.
- **References:** each case is compared with the serial direct run of the same physics. Without one
  (`--no-refs`, as at high resolution, where serial is too slow), it is compared with a partitioned
  direct run instead (`*_box_per`). Those agree with serial to about 1e-9 K.

## Reading the table

- **`T_sprd`, `Tbed_sprd`, `u_sprd`, `Zs_sprd`:** the largest range over a horizontal layer of T
  (K), basal T, u (relative) and the surface (m). The exact value is 0.
- **`v_max`:** max |v| / max |u|. The exact value is 0.
- **`dT_max`, `dT_iface`, `dT_inter`, `dT_side`:** |T − T_ref| everywhere, at partition-interface
  nodes, at interior nodes, and at nodes on the periodic sides.
- **`du_rel`:** the same comparison for u, relative to the reference's largest u.
- **`mismatch`:** partitions disagreeing about a shared node; it should be 0.
- **`worst_at`** (in `summary.csv`): where the largest deviation from the layer mean sits.

## Looking at it

Copy the small npz files, then run the plot scripts:

```bash
rsync -avz -e "ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de" --include='*/' --include='bed*.npz' \
  --include='summary.csv' --exclude='*' \
  slloyd@tethys-3g:/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates/periodic_test/out_sq/ \
  periodic_test/out_sq/
python periodic_test/plot_bed.py periodic_test/out_sq/np16           # final states, menu of cases
python periodic_test/plot_bed_time.py periodic_test/out_sq/np16      # transient cases, time slider
```

Both pages map the basal temperature minus its mean over the bed, the difference from the reference,
the surface minus its mean (FS cases), and the partitions. The partition interfaces are marked.
