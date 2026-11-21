#!/bin/bash
#SBATCH --output=./slurms/sticky_%j.out
#SBATCH --job-name=sticky
#SBATCH --ntasks=128
#SBATCH --time=06:00:00
# Sticky-air lava, step 1 (sticky/drop.sif): the footprint mesh by ElmerGrid, partitioned for the job's ranks, extruded
# by ElmerSolver; the run in sticky/run_<jobid>/. Needs the BFBt build (Elmer patch 04):
#   sbatch --ntasks=128 sticky/run.cmd          (ELMER_PREFIX defaults to /SCRATCH/$USER/elmer/install3)
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28} ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install3}
module load "$ELMER_MODULE"
export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
export ELMER_LIB=$ELMER_PREFIX/share/elmersolver/lib ELMER_SOLVER_HOME=$ELMER_PREFIX/share/elmersolver
export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/share/elmersolver/lib:$LD_LIBRARY_PATH
N=${SLURM_NTASKS:-1}
R=sticky/run_${SLURM_JOB_ID:-local}
mkdir -p $R && cp sticky/drop.sif sticky/sticky.lua sticky/box.grd $R/ && cd $R
ElmerGrid 1 2 box.grd -autoclean > grid.log 2>&1
[ "$N" -gt 1 ] && ElmerGrid 2 2 box -metiskway $N -nooverwrite >> grid.log 2>&1
echo drop.sif > ELMERSOLVER_STARTINFO
echo "$(date +%T) ElmerSolver on $N ranks in $R ($(command -v ElmerSolver))"
if [ "$N" -gt 1 ]; then mpirun -np $N --bind-to none ElmerSolver drop.sif > elmer.log 2>&1 ; else ElmerSolver drop.sif > elmer.log 2>&1 ; fi
echo "$(date +%T) done: $(tail -1 volume.dat 2>/dev/null)"
