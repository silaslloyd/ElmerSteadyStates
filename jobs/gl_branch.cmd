#!/bin/bash
#SBATCH --output=./slurms/gl_branch_%j.out
#SBATCH --job-name=glbranch
#SBATCH --ntasks=64
#SBATCH --time=2-00:00:00
#
# A branch from the converged S1 state with another grounding-line mesh resolution (jobs/gl_branch.py): does the
# state stay converged, and does the grounding line stay where it is? Run the 250 m control (the pipeline's own band: only
# the shelf cut and remesh) and finer ones, then compare the log lines "BRANCH <tag>: ...".
#   RES=250 sbatch jobs/gl_branch.cmd
#   RES=125 sbatch jobs/gl_branch.cmd
# CONFIG (default configs/flowline_gl_fine.toml), RES (m), ARGS (more options of gl_branch.py, e.g. "--max-chunks 60").
# Waits for nothing: it refuses to start before the pipeline's S1 is done (flags/s1_done) unless ARGS has --force.
# Resubmitting resumes.
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
unset ESS_SLIDING ESS_HEAT_LIMITER ESS_SEA_SPRING_DT ESS_STRAIN_HEAT_POSITIVE ESS_NO_CONVECTION
export ESS_CONTACT_FUNCTION=elmerstreams
python3 -u jobs/gl_branch.py --config ${CONFIG:-configs/flowline_gl_fine.toml} --res ${RES:-125} $ARGS
