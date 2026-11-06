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
#   CFG=configs/land_margin_3d.toml DT=50 YEARS=1000 FSSA=adapted FS_COUPLED=1 sbatch jobs/flowline_fssa.cmd   (3D)
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
CFG=${CFG:-configs/marine_flowline.toml}   # CFG=...: another case (3D tests: configs/land_margin_3d.toml, ...)
WORK=$(python3 -c "import os; from ess.config import load_config as L; print(os.path.relpath(L('$CFG')['_work']))")
GLARG=$(grep -q '^geometry *= *"marine"' $CFG && echo "--gl contact" || true)   # land: no grounding line
CASE=$(basename $CFG .toml); [ $CASE = marine_flowline ] && CASE= || CASE=${CASE}_   # tag prefix (not the flowline)
N=${SLURM_NTASKS:-32}
export ESS_NPROCS=$N ESS_MPIRUN="mpirun -np {n} --bind-to none"
# STOKES_PREC / STOKES_ELEMENT / LINEAR_STOKES: override the config's Stokes setup (3D tests: the configs' P2/P1 +
# ILU4 overflowed the ILU in 3D, and GCR + ILU0 stalled at a residual of 0.1; the defaults below for 3D cases: P1/P1 +
# bubbles (as the coarse model) with GCR + ILU2, the level the coarse runs settle at)
if [ -n "$CASE" ] && [ -z "$LINEAR_STOKES" ]; then
    # the coarse runs' Stokes solve (jobs/scaling_tests.cmd mkGCR): GCR(250) + ILU<level>, at most 2000 iterations,
    # tolerance 1e-8, carrying on when not converged
    STOKES_ELEMENT=${STOKES_ELEMENT:-Bubbles}; LINEAR_STOKES=block
    mkdir -p $WORK; GCRF=$WORK/stokes_gcr_${STOKES_PREC:-ILU2}_${SLURM_JOB_ID:-local}.sif.in   # per job (jobs side by side)
    printf '%s\n' "  Linear System Solver = Iterative" "  Linear System Iterative Method = GCR" \
        "  Linear System GCR Restart = Integer 250" "  Linear System Max Iterations = 2000" \
        "  Linear System Preconditioning = ${STOKES_PREC:-ILU2}" "  Linear System Convergence Tolerance = 1.0e-8" \
        "  Linear System Abort Not Converged = Logical False" "  Linear System Residual Output = 1" > $GCRF
    export ESS_STOKES_BLOCK_FILE=$GCRF
fi
[ -n "$STOKES_PREC" ] && export ESS_STOKES_PREC=$STOKES_PREC
[ -n "$STOKES_ELEMENT" ] && export ESS_STOKES_ELEMENT=$STOKES_ELEMENT
[ -n "$LINEAR_STOKES" ] && export ESS_LINEAR_STOKES=$LINEAR_STOKES
START=${START:-$WORK/fssa_start.npz}   # START=STATE: another start (e.g. a spun-up flowline state on this mesh)
( flock 9
  if [ ! -d $WORK/case/footprint/partitioning.$N ] || { [ ! -f $START ] && [ $START = $WORK/fssa_start.npz ]; }; then
      echo "$(date +%T) setting up $CFG for $N ranks"
      # ess init: a cheap, inexact Stokes solve (its velocity is not kept: the state is temperature and Zs)
      python3 -u -m ess setup $CFG && ESS_LINEAR_STOKES=block ESS_STOKES_BLOCK_FILE=elmer/stokes_cheap.sif.in \
          python3 -u -m ess init $CFG --out $WORK/fssa_start.npz
  fi ) 9> slurms/.fssa_setup_$(basename $CFG .toml).lock
DT=${DT:-10}; YEARS=${YEARS:-500}
STEPS=$(python3 -c "print(max(1, round($YEARS / $DT)))")
export ESS_FSSA=${FSSA:-full}
if [ "$FS_COUPLED" = 1 ]; then export ESS_FS_COUPLED=1; COUPLED_ITERS=${COUPLED_ITERS:-3}; fi
export ESS_COUPLED_ITERS=${COUPLED_ITERS:-1}
[ "$FSSA_PREV_TIMESTEP" = 1 ] && export ESS_FSSA_PREV_TIMESTEP=1 && TAG=${TAG:-${CASE}dt${DT}_adaptedts_coupled${ESS_COUPLED_ITERS}}
export ESS_STEP_CHECK=1 ESS_KEEP_RUNS=1 ESS_NO_CACHE=1 ESS_ELMER_TIMEOUT=604800
export ESS_VTU_EVERY=${VTU_EVERY:-$(python3 -c "print(max(1, $STEPS // 10))")}
TAG=${TAG:-${CASE}dt${DT}_${ESS_FSSA}$([ "$FS_COUPLED" = 1 ] && echo _coupled${ESS_COUPLED_ITERS} || true)}
echo "$(date +%T) FSSA test $TAG ($CFG): dt $DT yr x $STEPS steps, FSSA $ESS_FSSA, coupled free surface ${FS_COUPLED:-0}," \
     "coupled iterations $ESS_COUPLED_ITERS, VTU every $ESS_VTU_EVERY steps, $N ranks"
export ESS_RUN_LABEL=fssa${SLURM_JOB_ID:-local}
python3 -u -m ess run $CFG --state $START $GLARG --stokes-only --dt $DT --steps $STEPS \
    --out $WORK/fssa_$TAG.npz --output-level 4 > slurms/fssa_${SLURM_JOB_ID:-local}_$TAG.log 2>&1 \
    || echo "$(date +%T) run failed (log slurms/fssa_${SLURM_JOB_ID:-local}_$TAG.log)"
RUN=$(ls -d $WORK/runs/*_${ESS_RUN_LABEL}_*/ | head -1)
echo "run directory $RUN"
python3 periodic_test/step_check_summary.py $RUN || true
python3 plots/vtu_frames.py $RUN $WORK/fssa_${TAG}_frames.npz || true
grep 'MAIN: Time:' ${RUN}elmer.log | tail -1
echo "$(date +%T) done"
