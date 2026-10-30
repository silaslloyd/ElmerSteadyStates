#!/bin/bash
#SBATCH --output=./slurms/pgless_%j.out
#SBATCH --job-name="pgless"
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=12:00:00
#
# One step of the periodic grounding-line test on the ElmerStreams setup (periodic_test/README_gl_ess.md):
#   sbatch --ntasks=1 --time=00:20:00 periodic_test/run_gl_ess.cmd build   (the Elmer libraries only)
#   sbatch periodic_test/run_gl_ess.cmd prep [--from STATE]
#   sbatch periodic_test/run_gl_ess.cmd run fix|elmerstreams|none [--chunks N]   (prints its table at the end)
#   sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd analyse [MODE ...]
# Normally submitted by periodic_test/run_gl_ess.sh.
set -e
ARGS=("$@")
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
module load ${ELMER_MODULE:-elmerfem/2026-07-13}   # ELMER_MODULE=elmerfem sbatch ...: another build
if [ -n "$ELMER_PREFIX" ]; then        # a self-built Elmer (jobs/build_elmer.cmd) in front of the module's
    export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
    export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$LD_LIBRARY_PATH
fi
echo "Elmer: ${ELMER_MODULE:-elmerfem/2026-07-13} -> $(command -v ElmerSolver)"
build_libs() {
    local cur; cur=$(command -v ElmerSolver)
    if [ ! -f elmer/SteadyStateIO.so ] || [ ! -f elmer/Buoyancy.so ] || [ elmer/SteadyStateIO.F90 -nt elmer/SteadyStateIO.so ] \
       || [ ! -f elmer/RotateGuess.so ] || [ elmer/RotateGuess.F90 -nt elmer/RotateGuess.so ] \
       || [ ! -f elmer/GLDistance.so ] || [ elmer/GLDistance.F90 -nt elmer/GLDistance.so ] \
       || [ ! -f elmer/PeriodicGL.so ] || [ -n "$(find elmer -maxdepth 1 -name '*Periodic*.F90' -newer elmer/PeriodicGL.so)" ] \
       || [ ! -f elmer/ColdTempMask.so ] || [ ! -f elmer/SheetSolverhw.so ] \
       || [ "$(cat elmer/.built_with 2>/dev/null)" != "$cur" ]; then
        make -C elmer clean && make -C elmer && make -C elmer PeriodicGL.so thermal && echo "$cur" > elmer/.built_with
    fi
}
# run_gl_ess.sh builds once in its own job ("build") before the others: jobs on several nodes building at the
# same time remove each other's module files (flock does not hold across nodes on /SCRATCH)
( flock 9; build_libs ) 9> elmer/.build.lock
[ "${ARGS[0]}" = build ] && exit 0
python3 -u periodic_test/run_gl_ess.py "${ARGS[@]}"
