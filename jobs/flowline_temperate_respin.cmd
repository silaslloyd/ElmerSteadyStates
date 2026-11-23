#!/bin/bash
#SBATCH --output=./slurms/flowline_temperate_respin_%j.out
#SBATCH --job-name=flrespin
#SBATCH --ntasks=128
#SBATCH --time=3-00:00:00
#
# The temperate bound's spin-up of jobs/flowline_bounds.cmd (WHICH=temperate; extended strip, mechanics only, sliding
# temperate everywhere, grounding line by contact every step), repeated with another contact function and written
# to its own file. Why: with Elmer/Ice's stock SlidCoef_Contact (the gate relChange < Test Contact Tolerance) the spin-up
# stopped at a grounding line of 1039.25 km whose last grounded column was thinner than flotation (H/H_f 0.989,
# N/ref 0.52); with ElmerStreams' contact test the same state retreats and N/ref is 0.93.
#   sbatch jobs/flowline_temperate_respin.cmd
# Environment (all optional):
#   CONTACT       elmerice | elmerstreams (default elmerstreams)
#   TAG           name of the output, work/flowline_rect_ext/bound_temperate_<TAG>_spin.npz (default es)
#   START         the state to spin up from (default work/flowline_rect_ext/bed30.npz, as the original: grounding line 738 km)
#   SPIN_CHUNKS   chunks of 100 steps (default 50: with SPIN_DT 10 yr, 50 kyr like the stock-contact spin-up)
#   SPIN_DT       yr (default 10); SPIN_LINEAR  Stokes linear solver (default direct, as the original: the last pair of flowline_bounds jobs)
# Progress: python3 diag/cache_gl.py $(ls -t work/flowline_rect_ext/cache/*.npz | head -20)   (grounding line per saved chunk)
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
CFG=configs/flowline_rect_ext.toml
W=work/flowline_rect_ext
START=${START:-$W/bed30.npz}
TAG=${TAG:-es}
export ESS_SLIDING=temperate
unset ESS_FROZEN_FACTOR
export ESS_CONTACT_FUNCTION=${CONTACT:-elmerstreams}
OUT=$W/bound_temperate_${TAG}_spin.npz
echo "$(date +%T) temperate spin-up from $START, contact function $ESS_CONTACT_FUNCTION, ${SPIN_CHUNKS:-50} x 100 x ${SPIN_DT:-10} yr, Stokes ${SPIN_LINEAR:-direct} -> $OUT"
python3 -u -m ess run $CFG --state $START --stokes-only --gl contact --chunks ${SPIN_CHUNKS:-50} --steps 100 \
    --dt ${SPIN_DT:-10} --linear-stokes ${SPIN_LINEAR:-direct} --out $OUT
echo "$(date +%T) done"
