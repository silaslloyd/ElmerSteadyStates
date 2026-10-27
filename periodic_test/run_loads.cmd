#!/bin/bash
#SBATCH --output=./slurms/ploads_%j.out
#SBATCH --job-name="ploads"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#
# Periodic-sidewall loads test (periodic_test/README_loads.md). From the repository root:
#   sbatch periodic_test/run_loads.cmd                    # every case, then the table
#   sbatch periodic_test/run_loads.cmd --only ny1 fs_     # some cases
#   sbatch periodic_test/run_loads.cmd --element P2/P1    # the other Stokes element
set -e
ARGS=("$@")
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
module load elmerfem/2026-07-13
build_libs() {
    local cur; cur=$(command -v ElmerSolver)
    if [ ! -f elmer/SteadyStateIO.so ] || [ elmer/SteadyStateIO.F90 -nt elmer/SteadyStateIO.so ] \
       || [ "$(cat elmer/.built_with 2>/dev/null)" != "$cur" ]; then
        make -C elmer clean && make -C elmer && echo "$cur" > elmer/.built_with
    fi
}
( flock 9; build_libs ) 9> elmer/.build.lock
python3 -u periodic_test/run_loads.py --np "${SLURM_NTASKS:-4}" "${ARGS[@]}"
