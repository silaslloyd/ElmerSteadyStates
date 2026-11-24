#!/bin/bash
#SBATCH --output=./slurms/fssa3d_%j.out
#SBATCH --job-name="fssa3d"
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=04:00:00
#
# Standalone 3D implicit-vs-explicit free-surface test (fssa3d/make_case.py): full Stokes only, a slab with a friction
# dip across the width, a surface bump relaxing. Each case is serial (MUMPS), so up to --ntasks run side by side.
#
#   sbatch fssa3d/run.cmd                       # the default sweep below
#   YEARS=200 AMP=100 sbatch fssa3d/run.cmd
#   DTS="2 10 50" SCHEMES="explicit implicit" FSSAS="full none" sbatch fssa3d/run.cmd
#
# then: python3 fssa3d/analyse.py     (table + fssa3d/fssa3d.png)
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28}
export ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install2}
module load "$ELMER_MODULE"
if [ -n "$ELMER_PREFIX" ]; then
    export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
    export ELMER_LIB=$ELMER_PREFIX/share/elmersolver/lib ELMER_SOLVER_HOME=$ELMER_PREFIX/share/elmersolver
    export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/share/elmersolver/lib:$LD_LIBRARY_PATH
fi
YEARS=${YEARS:-100}
AMP=${AMP:-50}
DTS=${DTS:-"1 5 10 20 50"}
SCHEMES=${SCHEMES:-"explicit implicit"}
FSSAS=${FSSAS:-"full none"}
ITERS=${ITERS:-3}
EXTRA=${EXTRA:-}
N=${SLURM_NTASKS:-16}
export OMP_NUM_THREADS=1
JOBS=()
for s in $SCHEMES; do for f in $FSSAS; do for dt in $DTS; do
    JOBS+=("--tag ${s:0:3}_${f}_dt${dt} --scheme $s --fssa $f --dt $dt --years $YEARS --amp $AMP --iters $ITERS $EXTRA")
done; done; done
printf '%s\n' "${JOBS[@]}" | xargs -P "$N" -I{} sh -c 'python3 fssa3d/make_case.py {} --run || true'
python3 fssa3d/analyse.py || true
