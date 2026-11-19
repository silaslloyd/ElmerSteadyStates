#!/bin/bash
#SBATCH --output=./slurms/flowline_stability_%j.out
#SBATCH --job-name=flstab
#SBATCH --ntasks=64
#SBATCH --time=1-00:00:00
#
# Linear stability of a converged sweep point (ess stability): Arnoldi on the PHYSICAL map [stability_map]
# (100 x 50 yr, the bedrock's real heat capacity; Newton's map has it x 0.001) linearised at the state, M model runs
# (default 30, ~10 min each on the refined mesh). Both masks frozen in the map (the patch and the grounding line, as in
# Newton): the stability of the state with the patch and the grounding line held, not of the grounding line's motion.
# Logs the leading |mu| as Arnoldi proceeds; mu > 1 (sigma > 0) is unstable, sigma = ln(mu)/5000 yr its growth rate.
# Output work/flowline_rect_coarse/stability/<state name>/ (stability.npz: mu, sigma, modes; arnoldi.npz
# checkpoint: resubmitting resumes).
#   STATE=work/flowline_rect_coarse/sweep_refined/seg_m0245/points/state_07817.npz sbatch jobs/flowline_stability.cmd
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
[ -f $STATE ] || { echo "no state $STATE"; exit 1; }
echo "$(date +%T) stability of $STATE on $CFG, m = ${M:-30}"
python3 -u -m ess stability $CFG --state $STATE --m ${M:-30}
echo "$(date +%T) done"
