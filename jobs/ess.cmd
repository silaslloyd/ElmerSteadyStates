#!/bin/bash
#SBATCH --output=./slurms/ess_%j.out
#SBATCH --job-name="ESS"
#SBATCH --ntasks=32
#SBATCH --time=72:00:00
#SBATCH --mail-type=all
#
# Run one ElmerSteadyStates command in a batch job. The Python driver runs serially and launches
# `mpirun -np $SLURM_NTASKS ElmerSolver` for every map evaluation.
#
#   sbatch jobs/ess.cmd setup    configs/land_flowline.toml     (mesh partitioned for --ntasks)
#   sbatch jobs/ess.cmd init     configs/land_flowline.toml
#   sbatch jobs/ess.cmd newton   configs/land_flowline.toml --p 266
#   sbatch jobs/ess.cmd continue configs/land_flowline.toml --state work/land_flowline/steady_p266.npz
#   sbatch jobs/ess.cmd stability configs/land_flowline.toml --name Tsurf --point 12
#
# Use the same --ntasks for all jobs (the mesh partitioning is fixed by `setup`).
#
# Resubmitting the same command after a time-out resumes it (finished Elmer runs are cached,
# continuation picks up from the last point, Arnoldi from its checkpoint).

set -e
# save the arguments first: `module` / profile scripts can shift the positional parameters
ARGS=("$@")
if [ ${#ARGS[@]} -lt 2 ]; then
    echo "usage: sbatch jobs/ess.cmd <command> <config> [options]"; exit 1
fi
echo "ess ${ARGS[*]}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms

module load elmerfem/2026-07-13
export DIM=2

# the user libraries (state I/O, marine sea pressure) must be compiled against the Elmer that runs them: rebuild
# when that Elmer changes (elmer/.built_with records it), under a lock so simultaneous jobs build once
build_libs() {
    local cur; cur=$(command -v ElmerSolver)
    if [ ! -f elmer/SteadyStateIO.so ] || [ ! -f elmer/Buoyancy.so ] || [ elmer/SteadyStateIO.F90 -nt elmer/SteadyStateIO.so ] \
       || [ elmer/Buoyancy.F90 -nt elmer/Buoyancy.so ] || [ "$(cat elmer/.built_with 2>/dev/null)" != "$cur" ]; then
        echo "building elmer/*.so for $cur"
        make -C elmer clean && make -C elmer && echo "$cur" > elmer/.built_with
    fi
}
( flock 9; build_libs ) 9> elmer/.build.lock

python3 -u -m ess "${ARGS[@]}"
