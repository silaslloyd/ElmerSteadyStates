#!/bin/bash
#SBATCH --output=./slurms/periodic_%j.out
#SBATCH --job-name="ptest"
#SBATCH --ntasks=8
#SBATCH --time=04:00:00
#
# Periodic-BC / partitioning test (periodic_test/README.md). From the repository root:
#   sbatch periodic_test/run.cmd                  # every case
#   sbatch periodic_test/run.cmd --only A_ F_     # some groups
set -e
ARGS=("$@")
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
module load elmerfem
if [ ! -f elmer/SteadyStateIO.so ]; then
    make -C elmer
fi
python3 -u periodic_test/run_tests.py --np "${SLURM_NTASKS:-8}" "${ARGS[@]}"
