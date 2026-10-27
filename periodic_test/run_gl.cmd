#!/bin/bash
#SBATCH --output=./slurms/pgl_%j.out
#SBATCH --job-name="pgl"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=08:00:00
#
# Periodic grounding-line test (periodic_test/README_gl.md). From the repository root:
#   sbatch periodic_test/run_gl.cmd                     # old and new, serial, then the analysis
#   sbatch periodic_test/run_gl.cmd --only new --parts 4
#   sbatch periodic_test/run_gl.cmd --steps 50          # a short trial
set -e
ARGS=("$@")
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
module load elmerfem/2026-07-13
build_libs() {
    local cur; cur=$(command -v ElmerSolver)
    if [ ! -f elmer/SteadyStateIO.so ] || [ ! -f elmer/Buoyancy.so ] || [ elmer/SteadyStateIO.F90 -nt elmer/SteadyStateIO.so ] \
       || [ ! -f elmer/PeriodicGL.so ] || [ -n "$(find elmer -maxdepth 1 -name '*Periodic*.F90' -newer elmer/PeriodicGL.so)" ] \
       || [ "$(cat elmer/.built_with 2>/dev/null)" != "$cur" ]; then
        make -C elmer clean && make -C elmer && make -C elmer PeriodicGL.so && echo "$cur" > elmer/.built_with
    fi
}
( flock 9; build_libs ) 9> elmer/.build.lock
python3 -u periodic_test/run_gl.py "${ARGS[@]}"
