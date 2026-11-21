#!/bin/bash
#SBATCH --output=./slurms/flowline_sweep_branch_%j.out
#SBATCH --job-name=flbranch
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# A BRANCH of the refined marine sweep (jobs/sweep_refined.py), beside the main one (flowline_sweep_refined.cmd), to
# test speed-ups on the same points: from the converged point FROM_STATE (d = D_KM km; PREV_STATE the point before it,
# for the secant predictor), segments in work/flowline_rect_coarse/<OUT_ROOT>/ (default sweep_branch) with meshes
# tagged <TAG_PREFIX> (default b), PREDICTOR (default secant), on SLURM_NTASKS ranks (default 64: 128 ran each model run ~1.6x slower, 7.2 vs 4.5 min).
# The Newton fix of adb19f8 (the forcing term capped at eta_max) is in the code.
#   sbatch jobs/flowline_sweep_branch.cmd
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
W=work/flowline_rect_coarse
FROM_STATE=${FROM_STATE:-$W/sweep_refined/seg_m0495/points/state_05192.npz}
PREV_STATE=${PREV_STATE:-$W/sweep_refined/seg_m0495/points/state_05317.npz}
D_KM=${D_KM:-519.2}
echo "$(date +%T) branch sweep from $FROM_STATE (d $D_KM km, prev $PREV_STATE) on ${SLURM_NTASKS:-?} ranks, predictor ${PREDICTOR:-secant}"
python3 -u jobs/sweep_refined.py configs/flowline_rect_coarse.toml --from-state $FROM_STATE --d-km $D_KM \
    --prev-state $PREV_STATE --out-root ${OUT_ROOT:-sweep_branch} --tag-prefix ${TAG_PREFIX:-b} --d-end-km ${D_END:-0} \
    --span-km ${SPAN:-50} --pad-km ${PAD:-10} --fine ${FINE:-250} --step-frac ${STEP_FRAC:-0.2} \
    --step-min-km ${STEP_MIN:-1} --step-max-km ${STEP_MAX:-25} --predictor ${PREDICTOR:-secant}
echo "$(date +%T) done"
