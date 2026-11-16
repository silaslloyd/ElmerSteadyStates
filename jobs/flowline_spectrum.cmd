#!/bin/bash
#SBATCH --output=./slurms/flowline_spectrum_%j.out
#SBATCH --job-name=flspec
#SBATCH --ntasks=32
#SBATCH --time=3-00:00:00
#
# The marine mask sweep on the extended strip's flowline (configs/flowline_rect_ext.toml; ess spectrum, marine:
# ess/spectrum.py run_spectrum_marine): temperate patches from the grounding line upstream, d = 0, D_STEP, 2 D_STEP, ..
# km, each a Newton steady state with that cold/temperate mask frozen and the grounding line moved by flotation between
# solves, kept when no cold bed is above melting and the steady CS water sheet (the 3D runs' law) stays positive from
# the grounding line (h = hr) to the patch head. From the spin-up of jobs/flowline_steady.cmd (START, default
# work/flowline_rect_ext/spin.npz). Output: work/flowline_rect_ext/spectrum_p268.15/spectrum.csv and points/.
# Resubmitting resumes after the last row.
#   sbatch jobs/flowline_spectrum.cmd                     (D_STEP=5 km, stop after 3 failures in a row)
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
START=${START:-work/flowline_rect_ext/spin.npz}
echo "$(date +%T) marine mask sweep from $START: step ${D_STEP:-5} km, stop after ${STOP_AFTER:-3} failures in a row"
python3 -u -m ess spectrum configs/flowline_rect_ext.toml --state $START --d-step-km ${D_STEP:-5} \
    --stop-after-dry ${STOP_AFTER:-3} ${D_MAX:+--d-max-km $D_MAX}
