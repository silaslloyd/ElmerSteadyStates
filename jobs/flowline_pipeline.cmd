#!/bin/bash
#SBATCH --output=./slurms/flowline_pipeline_%j.out
#SBATCH --job-name=flpipe
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# The unattended marine flowline pipeline (jobs/flowline_pipeline.py): the SSA-solution start with its shelf, the
# temperate Stokes + free-surface transient to rms change < 1e-3, Newton, steady heat + Newton with the ct mask fixed,
# the bed sweep (50 km steps, a 500-yr Newton map, failed steps halved), and the hydrology and thermal bisectors
# submitted as their own jobs when the sweep has found a bracket. Everything with the ElmerStreams contact function.
# At most three of its jobs run at once: this one and the two bisectors. Every stage resumes; when this job's time
# runs out, its continuation (queued here, started only after this job ends) carries on; it stops by itself when
# <work_dir>/pipeline/flags/DONE or GAVEUP exists.
#   sbatch jobs/flowline_pipeline.cmd
# CONFIG: the case (default configs/flowline_rect_coarse.toml; configs/flowline_gl_fine.toml has the 250 m band at the
# corrected grounding line). Options for the driver in ARGS (e.g. ARGS="--until s3"); NO_CHAIN=1: no continuation job.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28} ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install2}
CONFIG=${CONFIG:-configs/flowline_rect_coarse.toml}; export CONFIG
WORK=$(python3 -c "import sys; sys.path.insert(0, '.'); from ess.config import load_config; print(load_config('$CONFIG')['_work'])")
FLAGS=$WORK/pipeline/flags
if [ -f $FLAGS/DONE ] || [ -f $FLAGS/GAVEUP ]; then echo "$(date +%T) pipeline already finished (flags): nothing to do"; exit 0; fi
if [ "${NO_CHAIN:-0}" != 1 ] && [ -n "$SLURM_JOB_ID" ]; then
    NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID jobs/flowline_pipeline.cmd) \
        && echo "$(date +%T) continuation job $NEXT queued (after $SLURM_JOB_ID)"
fi
bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
export ELMER_LIB=$ELMER_PREFIX/share/elmersolver/lib ELMER_SOLVER_HOME=$ELMER_PREFIX/share/elmersolver
export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/share/elmersolver/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export ESS_MPIRUN="mpirun -np {n} --bind-to none"
unset ESS_SLIDING ESS_HEAT_LIMITER ESS_SEA_SPRING_DT ESS_STRAIN_HEAT_POSITIVE ESS_NO_CONVECTION
export ESS_CONTACT_FUNCTION=elmerstreams
echo "$(date +%T) pipeline on ${SLURM_NTASKS:-?} ranks, contact function $ESS_CONTACT_FUNCTION"
python3 -u jobs/flowline_pipeline.py --config $CONFIG $ARGS
echo "$(date +%T) driver exited"
