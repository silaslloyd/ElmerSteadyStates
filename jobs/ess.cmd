#!/bin/bash
#SBATCH --output=./slurms/ess_%j.out
#SBATCH --job-name="ESS"
#SBATCH --ntasks=128
#SBATCH --time=72:00:00
#SBATCH --mail-type=all
#
# Run one ElmerSteadyStates command in a batch job. The Python driver runs serially and launches
# `mpirun -np $SLURM_NTASKS ElmerSolver` for every map evaluation.
#
#   sbatch jobs/ess.cmd continue configs/land_flowline.toml
#   sbatch jobs/ess.cmd newton   configs/land_flowline.toml --p 266
#   sbatch jobs/ess.cmd stability configs/land_flowline.toml --name Tsurf --point 12
#
# Resubmitting the same command after a time-out resumes it (finished Elmer runs are cached,
# continuation picks up from the last point, Arnoldi from its checkpoint).

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms

module load elmerfem
export DIM=2

if [ ! -f elmer/SteadyStateIO.so ]; then
    make -C elmer
fi

python3 -u -m ess "$@"
