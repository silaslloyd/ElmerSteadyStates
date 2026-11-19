#!/bin/bash
#SBATCH --output=./slurms/predictor_test_%j.out
#SBATCH --job-name=predtest
#SBATCH --ntasks=64
#SBATCH --time=1-00:00:00
#
# The sweep predictor test (jobs/predictor_test.py, ess/predictor.py): from the converged refined-margin point
# STATE (default d = 794.2 km of work/flowline_rect_coarse/sweep_refined/seg_m0245) to D_NEW km (default 769.2),
# F at the plain and at the shifted start (margin structure moved downstream with the head), then Newton from the
# shifted start (NEWTON=0: only the two residuals). On the segment's mesh (CFG, default its config). Output:
# work/flowline_rect_coarse/predictor_test_d<D_NEW>_s<SIGMA>/. SIGMA (km, default 15): only the anomaly about a
# Gaussian background of this width moved (0: whole profiles).
# MODE=secant PREV=earlier.npz: the secant start (linear extrapolation in d of PREV and STATE) instead.
#   sbatch jobs/predictor_test.cmd
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
CFG=${CFG:-work/flowline_rect_coarse/refined/m0245.toml}
STATE=${STATE:-work/flowline_rect_coarse/sweep_refined/seg_m0245/points/state_07942.npz}
D_NEW=${D_NEW:-769.2}
MODE=${MODE:-shift}
echo "$(date +%T) predictor test: $STATE -> d $D_NEW km on $CFG"
python3 -u jobs/predictor_test.py $CFG $STATE --d-new-km $D_NEW --mode $MODE ${PREV:+--prev $PREV} ${CORE:+--core-km $CORE} ${TAPER:+--taper-km $TAPER} ${SIGMA:+--sigma-km $SIGMA} \
    $([ "${NEWTON:-1}" = 1 ] && echo --newton)
echo "$(date +%T) done"
