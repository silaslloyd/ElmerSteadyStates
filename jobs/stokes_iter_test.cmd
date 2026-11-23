#!/bin/bash
#SBATCH --output=./slurms/stokes_iter_test_%j.out
#SBATCH --job-name=stkiter
#SBATCH --ntasks=64
#SBATCH --time=0-02:00:00
#
# Does iterative Stokes (P1 + bubbles, GCR + PREC) converge on a state? One map-like step (dt DT) of `ess run` with
# the solver overridden by environment only (the config untouched); the GCR residual history goes to the log.
# For the spanwise strip, where GCR + ILU0 stalled at a relative residual ~0.996 for 5000 iterations: the same
# solver on the FLOWLINE state (one element across, walls) separates the 3D strip (mesh, periodic sides) from the
# physics (the subtemperate sliding coefficient's 1e6 contrast, the shelf).
#   CFG=work/flowline_rect_coarse/refined/u0700.toml STATE=... PREC=ILU0 sbatch jobs/stokes_iter_test.cmd
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
: "${CFG:?CFG}" "${STATE:?STATE}"
export ESS_LINEAR_STOKES=iterative ESS_STOKES_PREC=${PREC:-ILU0} ESS_STOKES_ELEMENT=${ELEMENT:-Bubbles}
export ESS_RUN_LABEL=stkiter${SLURM_JOB_ID:-}
export ESS_KEEP_RUNS=1
echo "$(date +%T) Stokes test: $CFG, state $STATE, $ESS_STOKES_ELEMENT, GCR + $ESS_STOKES_PREC, dt ${DT:-50}"
python3 -u -m ess run $CFG --state $STATE --dt ${DT:-50} --steps 1 --chunks 1 --no-remask \
    --out work/flowline_rect_coarse/stokes_iter_test_${SLURM_JOB_ID:-local}.npz || echo "$(date +%T) the run failed"
D=$(ls -td work/flowline_rect_coarse/runs/*stkiter${SLURM_JOB_ID:-}* 2>/dev/null | head -1)
echo "run dir: $D"
[ -n "$D" ] && grep -E "gcr:|ERROR|ComputeChange" "$D/elmer.log" | head -40
echo "$(date +%T) done"
