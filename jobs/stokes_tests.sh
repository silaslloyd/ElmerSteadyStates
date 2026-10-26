#!/bin/bash
# Stokes linear-solver test: grounded (land) and marine flowlines x MUMPS (direct), GCR + ILU (iterative) and
# block preconditioning (block: elmer/stokes_block.sif.in), Stokes + free surface only, 32 ranks per run, as
# TWO batch jobs (one node each): A (128 tasks) prepares both meshes and runs marine x3 + land direct; B (64
# tasks) runs land iterative + land block and starts only after A has started. Run on the login node:
#
#   bash jobs/stokes_tests.sh
#
# Options by environment (passed to the jobs): CHUNKS (default 5), LAND_FROM / MARINE_FROM (start states),
# ESS_MPIRUN (default "mpirun -np {n} --bind-to none").   Results: python plots/compare_stokes_tests.py
set -e
cd "$(dirname "$0")/.."
for f in "${LAND_FROM:-work/land_flowline_T2k_strain/branch_temperate/temperate_steady.npz}" \
         "${MARINE_FROM:-work/marine_flowline/run_cold_p268.15.npz}"; do
    [ -f "$f" ] || { echo "start state $f not found"; exit 1; }
done
A=$(sbatch --parsable --export=ALL jobs/stokes_tests_A.cmd)
B=$(sbatch --parsable --export=ALL --dependency=after:$A jobs/stokes_tests_B.cmd)
echo "job A $A (128 tasks: preparation, marine direct/iterative/block, land direct)"
echo "job B $B (64 tasks: land iterative/block; starts after A has started)"
echo "logs: slurms/stokes_A_$A.out, slurms/stokes_B_$B.out and slurms/stokes_{A_$A,B_$B}_<geometry>_<solver>.log"
