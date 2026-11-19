#!/bin/bash
#SBATCH --output=./slurms/spread_run_%j.out
#SBATCH --job-name=spread
#SBATCH --ntasks=512
#SBATCH --time=2-00:00:00
#
# TEST CASE (branch test/lava-spreading): the viscous gravity current of configs/spread_ice.toml (a uniform 100 m cold
# layer on a flat bed, a central temperate source, zero-velocity walls, a no-slip bed, Arrhenius viscosity).
#   1. ess setup (mesh, partitioned for --ntasks) and ess init, if not done yet;
#   2. STATIC phase, the source OFF (--lua ess_src_a=0): STATIC_CHUNKS (default 10) chunks of STEPS (20) x DT (50) yr,
#      to check the cold layer stays put (the log's "rms change" per chunk should stay tiny); states in
#      work/spread_ice/static_chunks/points/, the end in work/spread_ice/static.npz (skipped when that exists);
#   3. the source ON from the end of the static phase: CHUNKS (default 50) chunks, model time carried on; states in
#      work/spread_ice/run_chunks/points/ (time stamps from the start of the static phase).
# START=file.npz with T0=its model time: phase 3 from that state instead (e.g. to go on after a time-out).
# STATIC_CHUNKS=0: no static phase. CFG=configs/spread_vent.toml: the vent version (work/spread_vent).
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
W=${W:-work/$(basename $CFG .toml)}            # the config's work_dir (configs/X.toml: work/X)
mkdir -p $W
if [ ! -d $W/case/footprint ]; then python3 -u -m ess setup $CFG; fi
if [ ! -f $W/state0.npz ]; then python3 -u -m ess init $CFG --out $W/state0.npz; fi
DTS=${DT:-50}; NS=${STEPS:-20}
if [ -z "$START" ]; then
    if [ "${STATIC_CHUNKS:-10}" = 0 ] && [ ! -f $W/static.npz ]; then
        cp $W/state0.npz $W/static.npz              # no static phase: the source on from the start
    fi
    if [ ! -f $W/static.npz ]; then
        echo "$(date +%T) static phase, the source off: ${STATIC_CHUNKS:-10} chunks of $NS x $DTS yr"
        python3 -u -m ess run $CFG --state $W/state0.npz --chunks ${STATIC_CHUNKS:-10} --steps $NS --dt $DTS \
            --lua ess_src_a=0.0 --gl frozen --no-remask --save-every 1 --out $W/static.npz
    fi
    START=$W/static.npz
    T0=$(python3 -c "import numpy as np; z = np.load('$START'); print(float(z['meta::time']) if 'meta::time' in z.files else 0.0)")
fi
echo "$(date +%T) source on from $START (t = ${T0:-0} yr): ${CHUNKS:-50} chunks of $NS x $DTS yr"
python3 -u -m ess run $CFG --state $START --chunks ${CHUNKS:-50} --steps $NS --dt $DTS \
    --gl frozen --no-remask --save-every 1 --time-offset ${T0:-0} --out $W/run.npz
echo "$(date +%T) done"
