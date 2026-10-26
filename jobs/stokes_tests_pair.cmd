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
SV=${1:?usage: sbatch jobs/stokes_tests_pair.cmd <direct|iterative|block|amg>[_cart] [prep]}
# amg: block preconditioning with BoomerAMG on the velocity block (elmer/stokes_block_amg.sif.in, Elmer with Hypre)
# <solver>_cart: the same with the bed condition in x/y/z components (--cartesian-bed; marine with the frozen
# grounding line), a test of the normal-tangential handling of the block solver (fixed in Elmer 2026-10-14)
MARINE_OPTS="--bed cold"
case $SV in
    *_cart) BASE=${SV%_cart}; CART_OPTS="--cartesian-bed"; MARINE_OPTS="--bed cold --gl frozen";;
    *) BASE=$SV; CART_OPTS="";;
esac
case $BASE in
    direct|iterative|block) SOLVER_OPTS="--linear-stokes $BASE $CART_OPTS";;
    amg) SOLVER_OPTS="--linear-stokes block --stokes-block-file elmer/stokes_block_amg.sif.in $CART_OPTS";;
    *) echo "unknown solver $SV (direct, iterative, block, amg, each optionally with _cart)"; exit 1;;
esac
PREP=${2:-}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
module load elmerfem/2026-07-13
export DIM=2
# the user libraries (state I/O, marine sea pressure) must be compiled against the Elmer that runs them: rebuild
# when that Elmer changes (elmer/.built_with records it), under a lock so simultaneous jobs build once
build_libs() {
    local cur; cur=$(command -v ElmerSolver)
    if [ ! -f elmer/SteadyStateIO.so ] || [ ! -f elmer/Buoyancy.so ] || [ elmer/SteadyStateIO.F90 -nt elmer/SteadyStateIO.so ] \
       || [ elmer/Buoyancy.F90 -nt elmer/Buoyancy.so ] || [ "$(cat elmer/.built_with 2>/dev/null)" != "$cur" ]; then
        echo "building elmer/*.so for $cur"
        make -C elmer clean && make -C elmer && echo "$cur" > elmer/.built_with
    fi
}
( flock 9; build_libs ) 9> elmer/.build.lock
export ESS_NPROCS=32
MPIRUN_DEFAULT='mpirun -np {n} --bind-to none'
export ESS_MPIRUN="${ESS_MPIRUN:-$MPIRUN_DEFAULT}"
CHUNKS=${CHUNKS:-1}
STEPS=${STEPS:-20}          # time steps per chunk (each config's own dt): a short solver test
LAND_FROM=${LAND_FROM:-work/land_flowline_T2k_strain/branch_temperate/temperate_steady.npz}
MARINE_FROM=${MARINE_FROM:-work/marine_flowline/run_cold_p268.15.npz}
L=slurms/stokes_${SV}_${SLURM_JOB_ID:-local}

if [ "$PREP" = prep ]; then
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
run marine $MARINE_OPTS
run land
fail=0
for p in $(jobs -p); do wait $p || fail=$((fail + 1)); done
echo "$(date +%T) done; $fail run(s) failed (see the logs)"
