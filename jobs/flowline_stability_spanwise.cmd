#!/bin/bash
#SBATCH --output=./slurms/flowline_stability_spanwise_%j.out
#SBATCH --job-name=flstabsp
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# Lateral (spanwise) stability of a converged flowline steady state on a wide 3D strip (jobs/stability_spanwise.py):
# FROM_STATE + D_KM, WIDTH_KM (100), NY (50 rows across, structured), WALLS=1 (free-slip walls; default periodic sides), FINE_X (1000 m along the subtemperate region), M (30),
# SOLVER (iterative), STOP_AFTER (mesh | newton: e.g. to time the wide strip first). Several nodes:
# sbatch --nodes=2 --ntasks=128. Resubmitting resumes.
#   FROM_STATE=... D_KM=339.2 sbatch jobs/flowline_stability_spanwise.cmd
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
: "${FROM_STATE:?FROM_STATE}" "${D_KM:?D_KM}"
echo "$(date +%T) spanwise stability of $FROM_STATE (d $D_KM km): ${WIDTH_KM:-100} km x ${NY:-50}, ${SLURM_NTASKS:-?} ranks"
python3 -u jobs/stability_spanwise.py configs/flowline_rect_coarse.toml --from-state $FROM_STATE --d-km $D_KM \
    --width-km ${WIDTH_KM:-100} --ny ${NY:-50} ${WALLS:+--walls} --fine-x ${FINE_X:-1000} --map-steps ${MAP_STEPS:-10} --m ${M:-30} \
    --solver ${SOLVER:-iterative} ${STOP_AFTER:+--stop-after $STOP_AFTER}
echo "$(date +%T) done"
