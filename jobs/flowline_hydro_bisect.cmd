#!/bin/bash
#SBATCH --output=./slurms/flowline_hydro_bisect_%j.out
#SBATCH --job-name=flhydbis
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# The boundary of hydrologically compatible steady states (the steady sheet reaches the patch head without running
# dry), by bisection in d between neighbouring converged sweep points (jobs/hydro_bisect.py); waits until the sweeps
# have such a pair. 500-yr Newton map (MAP_STEPS, default 10), tol NEWTON_TOL (default 1.25e-5), ROOTS (default the
# horizon sweep and the sweeps from head 700). Resubmitting resumes.
#   sbatch jobs/flowline_hydro_bisect.cmd
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
export ESS_NEWTON_TOL=${NEWTON_TOL:-1.25e-5}
W=work/flowline_rect_coarse
ROOTS=${ROOTS:-"$W/sweep_h500 $W/sweep_h500_700down $W/sweep_h500_700up $W/sia/t0700/sweep_cbet_map10"}
echo "$(date +%T) hydrology bisection over $ROOTS, map ${MAP_STEPS:-10} steps, tol $ESS_NEWTON_TOL"
python3 -u jobs/hydro_bisect.py configs/flowline_rect_coarse.toml $ROOTS --map-steps ${MAP_STEPS:-10}
echo "$(date +%T) done"
