#!/bin/bash
#SBATCH --output=./slurms/flowline_jump_%j.out
#SBATCH --job-name=fljump
#SBATCH --ntasks=64
#SBATCH --time=1-00:00:00
#
# JUMP to a distant point of the marine bed sweep (jobs/jump_refined.py): the patch head at HEAD_KM (default 600 km;
# or the length D_KM), from the most advanced converged point of the refined sweep (or FROM=state.npz): a refined mesh
# around the new head, the state transferred, a transient with the masks fixed until rms(F) < SWITCH (default 3e-2) or
# MAX_KYR (default 100 kyr), then Newton with the contact grounding-line rounds.
# Output work/flowline_rect_coarse/jump/j<head>/ (transient/state_NNNN.npz, transient_end.npz, sweep/points/).
# TRANSIENT_ONLY=1: stop after the transient. Resubmitting resumes.
#   sbatch jobs/flowline_jump.cmd
#   HEAD_KM=400 sbatch jobs/flowline_jump.cmd
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
if [ -n "$D_KM" ]; then TGT="--d-km $D_KM"; else TGT="--head-km ${HEAD_KM:-600}"; fi
echo "$(date +%T) jump: $TGT ${FROM:+from $FROM}"
python3 -u jobs/jump_refined.py configs/flowline_rect_coarse.toml $TGT ${FROM:+--from-state $FROM} \
    --switch ${SWITCH:-3e-2} --max-kyr ${MAX_KYR:-100} $([ "${TRANSIENT_ONLY:-0}" = 1 ] && echo --transient-only)
echo "$(date +%T) done"
