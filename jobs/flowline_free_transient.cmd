#!/bin/bash
#SBATCH --output=./slurms/flowline_free_transient_%j.out
#SBATCH --job-name=flfree
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# A free transient from a converged steady state, the ElmerStreams way (jobs/free_transient.py): contact grounding line
# every step, the cold/temperate mask moved after every chunk, heat solved, real bedrock heat capacity. On its own mesh
# (250 m from the patch head - 20 km to 1065 km). Resubmitting continues.
#   FROM_STATE (default the head-700 state), DT (5 yr), STEPS (40 per chunk: 200 yr), SAVE_EVERY (5 chunks: 1 kyr)
#   sbatch jobs/flowline_free_transient.cmd
# Progress: python3 jobs/free_transient.py configs/flowline_rect_coarse.toml --from-state FROM_STATE --report
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
FROM_STATE=${FROM_STATE:-work/flowline_rect_coarse/sweep_h500_700up/seg_u0700/points/state_03392.npz}
echo "$(date +%T) free transient from $FROM_STATE: dt ${DT:-5} yr x ${STEPS:-40} steps per chunk"
python3 -u jobs/free_transient.py configs/flowline_rect_coarse.toml --from-state $FROM_STATE --dt ${DT:-5} \
    --steps ${STEPS:-40} --save-every ${SAVE_EVERY:-5}
echo "$(date +%T) done"
