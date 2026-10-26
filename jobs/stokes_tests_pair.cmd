#!/bin/bash
#SBATCH --output=./slurms/stokes_%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#
# Stokes solver test, one node per solver (submit all three with jobs/stokes_tests.sh):
#   sbatch --job-name=stokes_<solver> jobs/stokes_tests_pair.cmd <solver> [prep]
# runs the land and the marine test with that Stokes linear solver (direct | iterative | block | amg) side by side,
# 32 ranks each (ESS_NPROCS), unpinned MPI ranks (ESS_MPIRUN) so the two Elmer runs share the node's cores.
# With "prep" it first builds both test meshes and start states (marker files .prep_done); without, it waits
# for those markers.
set -e
SV=${1:?usage: sbatch jobs/stokes_tests_pair.cmd <direct|iterative|block|amg> [prep]}
# amg: block preconditioning with BoomerAMG on the velocity block (elmer/stokes_block_amg.sif.in, Elmer with Hypre)
case $SV in
    direct|iterative|block) SOLVER_OPTS="--linear-stokes $SV";;
    amg) SOLVER_OPTS="--linear-stokes block --stokes-block-file elmer/stokes_block_amg.sif.in";;
    *) echo "unknown solver $SV (direct, iterative, block, amg)"; exit 1;;
esac
PREP=${2:-}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
module load elmerfem
export DIM=2
export ESS_NPROCS=32
MPIRUN_DEFAULT='mpirun -np {n} --bind-to none'
export ESS_MPIRUN="${ESS_MPIRUN:-$MPIRUN_DEFAULT}"
CHUNKS=${CHUNKS:-1}
STEPS=${STEPS:-20}          # time steps per chunk (each config's own dt): a short solver test
LAND_FROM=${LAND_FROM:-work/land_flowline_T2k_strain/branch_temperate/temperate_steady.npz}
MARINE_FROM=${MARINE_FROM:-work/marine_flowline/run_cold_p268.15.npz}
L=slurms/stokes_${SV}_${SLURM_JOB_ID:-local}

if [ "$PREP" = prep ]; then
    if [ ! -f elmer/SteadyStateIO.so ] || [ elmer/SteadyStateIO.F90 -nt elmer/SteadyStateIO.so ] || [ ! -f elmer/Buoyancy.so ]; then
        make -C elmer
    fi
    rm -f work/test_stokes_land/.prep_done work/test_stokes_marine/.prep_done
    for G in land marine; do
        C=configs/test_stokes_$G.toml; W=work/test_stokes_$G
        [ $G = land ] && FROM=$LAND_FROM || FROM=$MARINE_FROM
        echo "$(date +%T) prep $G (from $FROM)"
        python3 -u -m ess setup $C > ${L}_prep_$G.log 2>&1
        python3 -u -m ess init $C >> ${L}_prep_$G.log 2>&1
        python3 -u -m ess transfer $C --from "$FROM" --out $W/start.npz >> ${L}_prep_$G.log 2>&1
        touch $W/.prep_done
        echo "$(date +%T) prep $G done"
    done
else
    echo "$(date +%T) waiting for the preparation (work/test_stokes_{land,marine}/.prep_done)"
    for i in $(seq 1 360); do
        [ -f work/test_stokes_land/.prep_done ] && [ -f work/test_stokes_marine/.prep_done ] && break; sleep 10
    done
    [ -f work/test_stokes_land/.prep_done ] && [ -f work/test_stokes_marine/.prep_done ] \
        || { echo "no preparation after 1 h; stopping"; exit 1; }
fi

run() {   # run <geometry> [options]: in the background, log slurms/stokes_<solver>_<job>_<geometry>.log
    local G=$1; shift
    python3 -u -m ess run configs/test_stokes_$G.toml --state work/test_stokes_$G/start.npz --stokes-only \
        $SOLVER_OPTS --chunks $CHUNKS --steps $STEPS --out work/test_stokes_$G/run_$SV.npz "$@" > ${L}_$G.log 2>&1 &
    echo "$(date +%T) started $G $SV (pid $!, log ${L}_$G.log)"
}
run marine --bed cold
run land
fail=0
for p in $(jobs -p); do wait $p || fail=$((fail + 1)); done
echo "$(date +%T) done; $fail run(s) failed (see the logs)"
