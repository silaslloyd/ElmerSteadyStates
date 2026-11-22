#!/bin/bash
#SBATCH --output=./slurms/flowline_stability_fine_%j.out
#SBATCH --job-name=flstabf
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# Linear stability of the newest converged sweep point on a fine mesh over the whole subtemperate region and the
# grounding line (jobs/stability_fine.py): mesh, transfer, Newton re-converged there, then ess stability (the physical
# map, Arnoldi M products, default 30). FINE (m, default 500), FINE_FROM (km, default 150), FROM_STATE + D_KM (another
# source). Resubmitting resumes.
#   sbatch jobs/flowline_stability_fine.cmd
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
echo "$(date +%T) stability on a fine mesh: ${FINE:-500} m from ${FINE_FROM:-150} km, ${CT_FINE:-250} m at the head, m = ${M:-30}${FROM_STATE:+, from $FROM_STATE (d $D_KM)}"
python3 -u jobs/stability_fine.py configs/flowline_rect_coarse.toml --fine ${FINE:-500} --fine-from-km ${FINE_FROM:-150} \
    --ct-fine ${CT_FINE:-250} --m ${M:-30} ${TAG_SUFFIX:+--tag-suffix $TAG_SUFFIX} ${MAP_STEPS:+--map-steps $MAP_STEPS} ${FROM_STATE:+--from-state $FROM_STATE --d-km $D_KM}
echo "$(date +%T) done"
