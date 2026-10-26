#!/bin/bash
#SBATCH --output=./slurms/stokes_A_%j.out
#SBATCH --job-name="stokes_A"
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --time=24:00:00
#
# Stokes solver test, node A (submit with jobs/stokes_tests.sh): prepares both test meshes and start states,
# then runs marine direct / iterative / block and land direct side by side, 32 ranks each (ESS_NPROCS),
# unpinned MPI ranks (ESS_MPIRUN) so the four Elmer runs share the node's cores. Job B runs the other two land
# runs once the land preparation here is done (marker work/test_stokes_land/.prep_done).
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
module load elmerfem
export DIM=2
if [ ! -f elmer/SteadyStateIO.so ] || [ elmer/SteadyStateIO.F90 -nt elmer/SteadyStateIO.so ] || [ ! -f elmer/Buoyancy.so ]; then
    make -C elmer
fi
export ESS_NPROCS=32
MPIRUN_DEFAULT='mpirun -np {n} --bind-to none'
export ESS_MPIRUN="${ESS_MPIRUN:-$MPIRUN_DEFAULT}"
CHUNKS=${CHUNKS:-5}
LAND_FROM=${LAND_FROM:-work/land_flowline_T2k_strain/branch_temperate/temperate_steady.npz}
MARINE_FROM=${MARINE_FROM:-work/marine_flowline/run_cold_p268.15.npz}
L=slurms/stokes_A_${SLURM_JOB_ID:-local}
rm -f work/test_stokes_land/.prep_done work/test_stokes_marine/.prep_done

prep() {  # prep <geometry> <start state>
    local C=configs/test_stokes_$1.toml W=work/test_stokes_$1
    echo "$(date +%T) prep $1"
    python3 -u -m ess setup $C > ${L}_prep_$1.log 2>&1
    python3 -u -m ess init $C >> ${L}_prep_$1.log 2>&1
    python3 -u -m ess transfer $C --from "$2" --out $W/start.npz >> ${L}_prep_$1.log 2>&1
    touch $W/.prep_done
    echo "$(date +%T) prep $1 done ($W/start.npz)"
}
prep land "$LAND_FROM"
prep marine "$MARINE_FROM"

run() {   # run <geometry> <solver> [options]: in the background, log slurms/stokes_A_<job>_<geometry>_<solver>.log
    local G=$1 SV=$2; shift 2
    python3 -u -m ess run configs/test_stokes_$G.toml --state work/test_stokes_$G/start.npz --stokes-only \
        --linear-stokes $SV --chunks $CHUNKS --out work/test_stokes_$G/run_$SV.npz "$@" > ${L}_${G}_${SV}.log 2>&1 &
    echo "$(date +%T) started $G $SV (pid $!, log ${L}_${G}_${SV}.log)"
}
run marine direct --bed cold
run marine iterative --bed cold
run marine block --bed cold
run land direct
fail=0
for p in $(jobs -p); do wait $p || fail=$((fail + 1)); done
echo "$(date +%T) done; $fail run(s) failed (see the logs)"
