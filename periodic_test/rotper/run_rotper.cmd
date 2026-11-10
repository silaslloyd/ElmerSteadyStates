#!/bin/bash
#SBATCH --output=./slurms/rotper_%j.out
#SBATCH --job-name="rotper"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#
# The rotational periodic Stokes test on tethys: the same Elmer as the coarse runs, the ESS libraries built, then
# periodic_test/rotper/run_rotper.sh (three runs, the comparison).
#   sbatch periodic_test/rotper/run_rotper.cmd                (from the repo root)
#   ANGLE=45 sbatch periodic_test/rotper/run_rotper.cmd       (settings as run_rotper.sh: ANGLE, R0, R1, H, NP, ROTSIGN)
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28}
export ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install2}
bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
export ELMER_LIB=$ELMER_PREFIX/share/elmersolver/lib ELMER_SOLVER_HOME=$ELMER_PREFIX/share/elmersolver
export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/share/elmersolver/lib:$LD_LIBRARY_PATH
bash periodic_test/rotper/run_rotper.sh
