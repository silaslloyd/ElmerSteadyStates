#!/bin/bash
#SBATCH --output=./slurms/flowline_cold_transient_%j.out
#SBATCH --job-name=flcoldtr
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# A plain transient of the frozen-bed (no sliding where grounded) ice sheet, mechanics only (Stokes and the free
# surfaces; the temperature as the start state's, not solved), the grounding line by contact EVERY STEP, small steps,
# for as long as the job lasts. Each chunk prints its "rms change" (the state's change over the chunk: converged when it
# stops falling, ~1e-3 or below); the state is rewritten to OUT after every chunk and kept every SAVE_EVERY chunks in
# <OUT>_chunks/points/. Resubmitting continues from OUT.
#   DT (yr, default 5), STEPS per chunk (default 200: 1000 yr), SAVE_EVERY (default 5), CFG (default the cold start's
#   mesh, work/flowline_rect_coarse/refined/cold.toml), START (default work/flowline_rect_coarse/cold/start.npz),
#   OUT (default work/flowline_rect_coarse/cold/transient.npz), P (surface temperature parameter, default 258.15)
#   sbatch jobs/flowline_cold_transient.cmd
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
unset ESS_SLIDING ESS_HEAT_LIMITER ESS_SEA_SPRING_DT ESS_STRAIN_HEAT_POSITIVE ESS_NO_CONVECTION
W=work/flowline_rect_coarse
CFG=${CFG:-$W/refined/cold.toml}
START=${START:-$W/cold/start.npz}
OUT=${OUT:-$W/cold/transient.npz}
SAVE_EVERY=${SAVE_EVERY:-5}
OFF=0
if [ -f "$OUT" ]; then
    START=$OUT
    OFF=$(( $(ls ${OUT%.npz}_chunks/points/state_*.npz 2>/dev/null | wc -l) * SAVE_EVERY ))
fi
echo "$(date +%T) frozen-bed transient, mechanics only, contact grounding line every step: ${DT:-5} yr x ${STEPS:-200} per chunk, from $START (chunk offset $OFF)"
ESS_SLIDING=frozen python3 -u -m ess run $CFG --state $START --p ${P:-258.15} --stokes-only --bed cold --gl contact \
    --dt ${DT:-5} --steps ${STEPS:-200} --chunks 100000 --save-every $SAVE_EVERY --chunk-offset $OFF --out $OUT
echo "$(date +%T) done"
