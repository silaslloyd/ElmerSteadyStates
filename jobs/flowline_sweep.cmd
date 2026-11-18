#!/bin/bash
#SBATCH --output=./slurms/flowline_sweep_%j.out
#SBATCH --job-name=flsweep
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# The marine bed sweep on the coarse flowline (configs/flowline_rect_coarse.toml; ess sweep, ess/spectrum.py
# run_sweep_marine): steady states with a temperate patch of length d from the grounding line upstream (cold
# upstream of it), d from D_START (default: the whole grounded bed, fully temperate) to D_END (default 0, all cold)
# in steps of STEP_FRAC x d (default 0.2) between STEP_MIN and STEP_MAX km (default 1 and 50), each from the previous
# state; Newton with the masks frozen, the grounding line moved by contact between solves. Nothing is rejected (the
# counts of cold nodes above melting and temperate nodes refreezing are recorded). From START (default the fully
# temperate steady state of jobs/flowline_coarse_newton.cmd). Output: work/flowline_rect_coarse/sweep_p268.15_down/
# (sweep.csv, points/). Resubmitting resumes after the last converged length.
#   sbatch jobs/flowline_sweep.cmd
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
CFG=configs/flowline_rect_coarse.toml
START=${START:-work/flowline_rect_coarse/steady_temperate.npz}
[ -f $START ] || { echo "no start state $START (jobs/flowline_coarse_newton.cmd first)"; exit 1; }
echo "$(date +%T) sweep from $START: d ${D_START:-whole grounded bed} -> ${D_END:-0} km, steps ${STEP_FRAC:-0.2} x d in ${STEP_MIN:-1} .. ${STEP_MAX:-50} km"
python3 -u -m ess sweep $CFG --state $START ${D_START:+--d-start-km $D_START} --d-end-km ${D_END:-0} \
    --step-frac ${STEP_FRAC:-0.2} --step-min-km ${STEP_MIN:-1} --step-max-km ${STEP_MAX:-50} ${OUT:+--out $OUT}
echo "$(date +%T) done"
