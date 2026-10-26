#!/bin/bash
# Stokes linear-solver test: grounded (land) and marine flowlines x MUMPS (direct), GCR + ILU (iterative) and
# block preconditioning (block: elmer/stokes_block.sif.in), Stokes + free surface only, 32 ranks per run, as
# THREE batch jobs (one node each, 64 tasks), one per solver, each running the land and the marine test side
# by side. The direct job first builds both meshes and start states; the other two start after it has
# started and wait for that. Run on the login node:
#
#   bash jobs/stokes_tests.sh
#
# Options by environment (passed to the jobs): STEPS (time steps per run, default 20), CHUNKS (default 1), LAND_FROM / MARINE_FROM (start states),
# ESS_MPIRUN (default "mpirun -np {n} --bind-to none").   Results: python plots/compare_stokes_tests.py
set -e
cd "$(dirname "$0")/.."
for f in "${LAND_FROM:-work/land_flowline_T2k_strain/branch_temperate/temperate_steady.npz}" \
         "${MARINE_FROM:-work/marine_flowline/run_cold_p268.15.npz}"; do
    [ -f "$f" ] || { echo "start state $f not found"; exit 1; }
done
D=$(sbatch --parsable --export=ALL --job-name=stokes_direct jobs/stokes_tests_pair.cmd direct prep)
I=$(sbatch --parsable --export=ALL --job-name=stokes_iterative --dependency=after:$D jobs/stokes_tests_pair.cmd iterative)
K=$(sbatch --parsable --export=ALL --job-name=stokes_block --dependency=after:$D jobs/stokes_tests_pair.cmd block)
echo "direct:    job $D (builds the meshes and start states, then land + marine)"
echo "iterative: job $I (land + marine, after the preparation)"
echo "block:     job $K (land + marine, after the preparation)"
echo "logs: slurms/stokes_<solver>_<job>.out and slurms/stokes_<solver>_<job>_{prep_land,prep_marine,land,marine}.log"
