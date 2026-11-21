#!/bin/bash
#SBATCH --output=./slurms/flowline_sia_%j.out
#SBATCH --job-name=flsia
#SBATCH --ntasks=64
#SBATCH --time=1-00:00:00
#
# A Newton start from the user's dimensionless SIA steady profile (jobs/sia_start.py): scaled as plot_section --sia's
# "temperate fit" against the refined sweep's newest point (or FROM=state.npz), a refined mesh around the SIA's
# cold/temperate transition (the patch head), the state transferred, the grounded surface set to the SIA's, steady heat
# on that geometry (ess run --heat-only, HEAT_STEPS x HEAT_DT yr), then Newton with the contact grounding-line rounds.
# Output work/flowline_rect_coarse/sia/s<head>/ (start_sia.npz, start_heat.npz, sweep/). NO_NEWTON=1: stop after the
# heat solve. Resubmitting resumes.
#   sbatch jobs/flowline_sia_start.cmd
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
echo "$(date +%T) sia start ${FROM:+from $FROM}"
python3 -u jobs/sia_start.py configs/flowline_rect_coarse.toml ${FROM:+--from-state $FROM} \
    --heat-dt ${HEAT_DT:-1e4} --heat-steps ${HEAT_STEPS:-30} $([ "${COLD_START:-1}" = 0 ] && echo --no-cold-start) $([ "${BED_SIA:-1}" = 0 ] && echo --no-bed-sia) $([ "${NO_NEWTON:-0}" = 1 ] && echo --no-newton)
echo "$(date +%T) done"
