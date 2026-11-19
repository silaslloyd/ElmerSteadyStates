#!/bin/bash
#SBATCH --output=./slurms/spread_run_%j.out
#SBATCH --job-name=spread
#SBATCH --ntasks=64
#SBATCH --time=2-00:00:00
#
# TEST CASE (branch test/lava-spreading): the cold viscous gravity current of configs/spread_ice.toml (a uniform cold
# layer on a flat bed, a central source, free-slip walls, a no-slip bed, Arrhenius viscosity), time-stepped.
#   1. ess setup (mesh, partitioned for --ntasks) and ess init, if not done yet;
#   2. ess run: CHUNKS (default 50) chunks of STEPS (default 20) x DT (default 50) yr, masks frozen (all grounded,
#      all cold), the state after every chunk in work/spread_ice/run_chunks/points/ (state_NNNN.npz).
# START=file.npz: go on from a saved state (e.g. the last chunk), with T0=its model time for the time stamps.
#   sbatch jobs/spread_run.cmd
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28} ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install2}
bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
export ELMER_LIB=$ELMER_PREFIX/share/elmersolver/lib ELMER_SOLVER_HOME=$ELMER_PREFIX/share/elmersolver
export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/share/elmersolver/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export ESS_MPIRUN="mpirun -np {n} --bind-to none"
CFG=${CFG:-configs/spread_ice.toml}
W=work/spread_ice
mkdir -p $W
if [ ! -d $W/case/footprint ]; then python3 -u -m ess setup $CFG; fi
if [ ! -f $W/state0.npz ]; then python3 -u -m ess init $CFG --out $W/state0.npz; fi
START=${START:-$W/state0.npz}
echo "$(date +%T) spreading run from $START: ${CHUNKS:-50} chunks of ${STEPS:-20} x ${DT:-50} yr"
python3 -u -m ess run $CFG --state $START --chunks ${CHUNKS:-50} --steps ${STEPS:-20} --dt ${DT:-50} \
    --gl frozen --no-remask --save-every 1 --time-offset ${T0:-0} --out $W/run.npz
echo "$(date +%T) done"
