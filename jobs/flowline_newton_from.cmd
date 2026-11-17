#!/bin/bash
#SBATCH --output=./slurms/flowline_newton_%j.out
#SBATCH --job-name=flnewton
#SBATCH --ntasks=128
#SBATCH --time=3-00:00:00
#
# Mechanics-only Newton on the extended strip's flowline (configs/flowline_rect_ext.toml) from a given state, with the
# bed temperate or frozen (as jobs/flowline_bounds.cmd's Newton step). WHICH=temperate|cold: the bound; STATE: the start
# (default: the newest finished chunk of that bound in the run cache, jobs/state_from_cache.py ->
# work/flowline_rect_ext/bound_<which>_now.npz); OUT (default work/flowline_rect_ext/bound_<which>_newton.npz).
#   WHICH=temperate sbatch jobs/flowline_newton_from.cmd
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
WHICH=${WHICH:-temperate}
case $WHICH in cold) export ESS_SLIDING=frozen;; temperate) export ESS_SLIDING=temperate;; *) echo "WHICH: cold | temperate"; exit 1;; esac
if [ -z "$STATE" ]; then
    STATE=$W/bound_${WHICH}_now.npz
    python3 jobs/state_from_cache.py $W $W/bed30.npz $STATE --which $WHICH
fi
OUT=${OUT:-$W/bound_${WHICH}_newton.npz}
echo "$(date +%T) $WHICH bound: Newton (mechanics only, sliding $ESS_SLIDING) from $STATE -> $OUT"
python3 -u -m ess newton $CFG --state $STATE --stokes-only --out $OUT
echo "$(date +%T) done"
