#!/bin/bash
#SBATCH --output=./slurms/fssa_%j.out
#SBATCH --job-name="ESS-FSSA"
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=04:00:00
#
# Free-surface time stepping on the marine flowline (configs/marine_flowline.toml: the ElmerStreams Marine_Term
# geometry on a strip one element wide; contact grounding line, full Stokes shelf, Stokes and the free surfaces only,
# the temperature frozen): how big a step each FSSA variant takes.
#
#   DT=10 YEARS=500 FSSA=full sbatch jobs/flowline_fssa.cmd
#   DT=10 YEARS=500 FSSA=adapted FS_COUPLED=1 COUPLED_ITERS=3 sbatch jobs/flowline_fssa.cmd
#
# DT (yr, default 10), YEARS (default 500), FSSA = full | normal | transposed | none | adapted ([elmer] fssa),
# FS_COUPLED=1: mesh mapping + free surfaces in the coupled loop with Stokes, COUPLED_ITERS (default 1; 3 with
# FS_COUPLED), VTU_EVERY (default: 10 outputs), TAG (default from the settings). Every run: the step check (CFL,
# change per step, relaxation number), boundary VTUs (plots/vtu_frames.py), the run directory kept. The first job
# builds the mesh for its --ntasks and the start state work/marine_flowline/fssa_start.npz (`ess init`: the
# Marine_Term initial geometry); later jobs reuse them (same --ntasks).
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28}
export ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install2}
bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
if [ -n "$ELMER_PREFIX" ]; then        # the self-built Elmer in front of the module's (as jobs/scaling_tests.cmd)
    export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
    export ELMER_LIB=$ELMER_PREFIX/share/elmersolver/lib ELMER_SOLVER_HOME=$ELMER_PREFIX/share/elmersolver
    export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/share/elmersolver/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
fi
CFG=configs/marine_flowline.toml
WORK=work/marine_flowline
N=${SLURM_NTASKS:-32}
export ESS_NPROCS=$N ESS_MPIRUN="mpirun -np {n} --bind-to none"
START=$WORK/fssa_start.npz
( flock 9
  if [ ! -f $START ] || [ ! -d $WORK/case/footprint/partitioning.$N ]; then
      echo "$(date +%T) setting up $CFG for $N ranks"
      python3 -u -m ess setup $CFG && python3 -u -m ess init $CFG --out $START
  fi ) 9> slurms/.fssa_setup.lock
DT=${DT:-10}; YEARS=${YEARS:-500}
STEPS=$(python3 -c "print(max(1, round($YEARS / $DT)))")
export ESS_FSSA=${FSSA:-full}
[ "$FS_COUPLED" = 1 ] && export ESS_FS_COUPLED=1 && COUPLED_ITERS=${COUPLED_ITERS:-3}
export ESS_COUPLED_ITERS=${COUPLED_ITERS:-1}
export ESS_STEP_CHECK=1 ESS_KEEP_RUNS=1 ESS_NO_CACHE=1 ESS_ELMER_TIMEOUT=604800
export ESS_VTU_EVERY=${VTU_EVERY:-$(python3 -c "print(max(1, $STEPS // 10))")}
TAG=${TAG:-dt${DT}_${ESS_FSSA}$([ "$FS_COUPLED" = 1 ] && echo _coupled${ESS_COUPLED_ITERS})}
echo "$(date +%T) flowline FSSA test $TAG: dt $DT yr x $STEPS steps, FSSA $ESS_FSSA, coupled free surface ${FS_COUPLED:-0}," \
     "coupled iterations $ESS_COUPLED_ITERS, VTU every $ESS_VTU_EVERY steps, $N ranks"
export ESS_RUN_LABEL=fssa${SLURM_JOB_ID:-local}
python3 -u -m ess run $CFG --state $START --gl contact --stokes-only --dt $DT --steps $STEPS \
    --out $WORK/fssa_$TAG.npz --output-level 4 > slurms/fssa_${SLURM_JOB_ID:-local}_$TAG.log 2>&1 \
    || echo "$(date +%T) run failed (log slurms/fssa_${SLURM_JOB_ID:-local}_$TAG.log)"
RUN=$(ls -d $WORK/runs/*_${ESS_RUN_LABEL}_*/ | head -1)
echo "run directory $RUN"
python3 periodic_test/step_check_summary.py $RUN || true
python3 plots/vtu_frames.py $RUN $WORK/fssa_${TAG}_frames.npz || true
grep 'MAIN: Time:' ${RUN}elmer.log | tail -1
echo "$(date +%T) done"
