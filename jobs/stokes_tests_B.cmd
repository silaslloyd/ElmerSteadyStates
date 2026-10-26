#!/bin/bash
#SBATCH --output=./slurms/stokes_B_%j.out
#SBATCH --job-name="stokes_B"
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#
# Stokes solver test, node B (submit with jobs/stokes_tests.sh, after job A has started): waits until job A has
# prepared the land mesh and start state, then runs land iterative and land block side by side, 32 ranks each.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
module load elmerfem
export DIM=2
export ESS_NPROCS=32
MPIRUN_DEFAULT='mpirun -np {n} --bind-to none'
export ESS_MPIRUN="${ESS_MPIRUN:-$MPIRUN_DEFAULT}"
CHUNKS=${CHUNKS:-5}
L=slurms/stokes_B_${SLURM_JOB_ID:-local}
echo "$(date +%T) waiting for the land preparation of job A (work/test_stokes_land/.prep_done)"
for i in $(seq 1 360); do [ -f work/test_stokes_land/.prep_done ] && break; sleep 10; done
[ -f work/test_stokes_land/.prep_done ] || { echo "no land preparation after 1 h; stopping"; exit 1; }

run() {   # run <geometry> <solver>
    local G=$1 SV=$2; shift 2
    python3 -u -m ess run configs/test_stokes_$G.toml --state work/test_stokes_$G/start.npz --stokes-only \
        --linear-stokes $SV --chunks $CHUNKS --out work/test_stokes_$G/run_$SV.npz "$@" > ${L}_${G}_${SV}.log 2>&1 &
    echo "$(date +%T) started $G $SV (pid $!, log ${L}_${G}_${SV}.log)"
}
run land iterative
run land block
fail=0
for p in $(jobs -p); do wait $p || fail=$((fail + 1)); done
echo "$(date +%T) done; $fail run(s) failed (see the logs)"
