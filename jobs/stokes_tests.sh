#!/bin/bash
# Stokes linear-solver test: grounded (land) and marine flowlines, each with MUMPS (direct), GCR + ILU
# (iterative) and block preconditioning (block, keywords in elmer/stokes_block.sif.in): Stokes + free surface
# only (ess run --stokes-only), 32 MPI ranks per run, the same start state and steps for the three solvers.
# Run on the login node (it only submits):
#
#   bash jobs/stokes_tests.sh                      Slurm places the runs (each on one node)
#   bash jobs/stokes_tests.sh NODE_A NODE_B        4 runs on NODE_A (marine x3 + land direct), 2 on NODE_B
#                                                  (land iterative, land block); free nodes: sinfo -N -l
# Options by environment: CHUNKS (default 5), LAND_FROM / MARINE_FROM (start states, any mesh: carried over
# with ess transfer), N (ranks, default 32).
#
# Then, when they have finished:   python plots/compare_stokes_tests.py
set -e
cd "$(dirname "$0")/.."
N=${N:-32}
CHUNKS=${CHUNKS:-5}
LAND_FROM=${LAND_FROM:-work/land_flowline_T2k_strain/branch_temperate/temperate_steady.npz}
MARINE_FROM=${MARINE_FROM:-work/marine_flowline/run_cold_p268.15.npz}
NODE_A=${1:-}
NODE_B=${2:-}
for f in "$LAND_FROM" "$MARINE_FROM"; do [ -f "$f" ] || { echo "start state $f not found"; exit 1; }; done

sub() {   # sub <node or ""> <dependency or ""> <ess arguments...>: prints the job id
    local node=$1 dep=$2; shift 2
    sbatch --parsable --nodes=1 --ntasks=$N ${node:+--nodelist=$node} ${dep:+--dependency=afterok:$dep} \
        --job-name=stokes_test jobs/ess.cmd "$@"
}

prep() {  # prep <geometry> <start state>: mesh, initial state, the start carried over; prints the last job id
    local C=configs/test_stokes_$1.toml W=work/test_stokes_$1
    local J1 J2
    J1=$(sub "" "" setup $C)
    J2=$(sub "" "$J1" init $C)
    sub "" "$J2" transfer $C --from "$2" --out $W/start.npz
}

run() {   # run <geometry> <solver> <node> <dependency> [extra ess run options]
    local G=$1 SV=$2 node=$3 dep=$4; shift 4
    local C=configs/test_stokes_$G.toml W=work/test_stokes_$G
    local J
    J=$(sub "$node" "$dep" run $C --state $W/start.npz --stokes-only --linear-stokes $SV --chunks $CHUNKS \
        --out $W/run_$SV.npz "$@")
    echo "  $G $SV: job $J${node:+ on $node}"
}

echo "preparing the meshes and start states (land from $LAND_FROM, marine from $MARINE_FROM)"
PL=$(prep land "$LAND_FROM")
PM=$(prep marine "$MARINE_FROM")
echo "  land: transfer job $PL; marine: transfer job $PM"
echo "the runs ($CHUNKS chunks each, $N ranks):"
run marine direct    "$NODE_A" "$PM" --bed cold
run marine iterative "$NODE_A" "$PM" --bed cold
run marine block     "$NODE_A" "$PM" --bed cold
run land   direct    "$NODE_A" "$PL"
run land   iterative "$NODE_B" "$PL"
run land   block     "$NODE_B" "$PL"
echo "progress: squeue -u $USER -n stokes_test; logs in slurms/ess_<job>.out"
echo "results:  python plots/compare_stokes_tests.py"
