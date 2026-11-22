#!/bin/bash
#SBATCH --output=./slurms/flowline_cold_start_%j.out
#SBATCH --job-name=flcold
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# The subtemperate all-cold-bed steady state at 268.15 K reached from a frozen-bed sheet by warming the surface
# (jobs/cold_start.py): frozen-bed spin-up and Newton at TC (default 258.15), the steady heat there, full Newton,
# continuation in Tsurf up to TEND (default 268.15), Newton at TEND. Output work/flowline_rect_coarse/cold/,
# work/flowline_rect_coarse/branch_cold/. FROM=state.npz (default the coarse sweep's last point), SPIN (chunks,
# default 5), STOP_AFTER (1-6). Resubmitting resumes.
#   sbatch jobs/flowline_cold_start.cmd
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
export ESS_NEWTON_TOL=${NEWTON_TOL:-5e-5}
echo "$(date +%T) cold start ${FROM:+from $FROM}, Newton tol $ESS_NEWTON_TOL"
python3 -u jobs/cold_start.py configs/flowline_rect_coarse.toml ${FROM:+--from-state $FROM} --tc ${TC:-258.15} \
    --tend ${TEND:-268.15} --spin ${SPIN:-5} --stop-after ${STOP_AFTER:-6}
echo "$(date +%T) done"
