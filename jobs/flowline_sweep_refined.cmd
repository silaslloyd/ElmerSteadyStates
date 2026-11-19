#!/bin/bash
#SBATCH --output=./slurms/flowline_sweep_refined_%j.out
#SBATCH --job-name=flsweepr
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# The marine bed sweep with the mesh refined at the cold/temperate margin (jobs/sweep_refined.py): in segments of
# SPAN km (default 50) of patch length, each on its own mesh of configs/flowline_rect_coarse.toml with a FINE m band
# (default 250) around the heads the segment covers (PAD km either side, default 10, graded over 10 km), set up and
# transferred from the previous segment's last converged state. Starts from the last converged point of FROM_SWEEP
# (default the coarse sweep, work/flowline_rect_coarse/sweep_p268.15_down). Steps STEP_FRAC x d (default 0.2) in
# STEP_MIN .. STEP_MAX km (default 1 .. 25), down to D_END km (default 0).
# Output: work/flowline_rect_coarse/sweep_refined/seg_mXXXX/ (sweep.csv, points/; XXXX the segment's first head in km),
# the segment configs work/flowline_rect_coarse/refined/mXXXX.toml, meshes work/flowline_rect_coarse/mXXXX/.
# PREDICTOR=secant: each Newton from the linear extrapolation of the last two converged points. Resubmitting resumes.
#   sbatch jobs/flowline_sweep_refined.cmd
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
FROM_SWEEP=${FROM_SWEEP:-work/flowline_rect_coarse/sweep_p268.15_down}
echo "$(date +%T) refined sweep from the last converged point of $FROM_SWEEP: segments of ${SPAN:-50} km, band ${FINE:-250} m"
python3 -u jobs/sweep_refined.py configs/flowline_rect_coarse.toml --from-sweep $FROM_SWEEP --d-end-km ${D_END:-0} \
    --span-km ${SPAN:-50} --pad-km ${PAD:-10} --fine ${FINE:-250} --step-frac ${STEP_FRAC:-0.2} \
    --step-min-km ${STEP_MIN:-1} --step-max-km ${STEP_MAX:-25} --predictor ${PREDICTOR:-none}
echo "$(date +%T) done"
