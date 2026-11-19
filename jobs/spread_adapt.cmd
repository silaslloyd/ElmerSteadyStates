#!/bin/bash
#SBATCH --output=./slurms/spread_adapt_%j.out
#SBATCH --job-name=spradapt
#SBATCH --ntasks=512
#SBATCH --time=2-00:00:00
#
# TEST CASE (branch test/lava-spreading): the spreading run with the mesh following the current's margin
# (jobs/spread_adapt.py, ess/spread_remesh.py). CFG (default configs/spread_vent.toml), from START (default the work
# dir's static.npz) at T0 (default its time), CHUNKS (50) of STEPS (200) x DT (5) yr; a new mesh fitted to the start
# first (REMESH_FIRST=0: not), then whenever the margin leaves its fine band. The remeshing settings: [case]
# spread_remesh_fine (2000 m), _band_km (10), _grade (0.2), _coarse_in (2 x spread_res), _coarse_out (spread_res),
# _trigger (1.5). States in <work>/adapt/points/, meshes <work>/case/a001, a002, ...
#   CFG=configs/spread_vent.toml START=work/spread_vent/run_chunks/points/state_0003.npz sbatch jobs/spread_adapt.cmd
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
CFG=${CFG:-configs/spread_vent.toml}
W=${W:-work/$(basename $CFG .toml)}
START=${START:-$W/static.npz}
echo "$(date +%T) adaptive spreading run: $CFG from $START"
python3 -u jobs/spread_adapt.py $CFG --state $START ${T0:+--t0 $T0} --chunks ${CHUNKS:-50} --steps ${STEPS:-200} \
    --dt ${DT:-5} --out $W/adapt $([ "${REMESH_FIRST:-1}" = 1 ] && echo --remesh-first)
echo "$(date +%T) done"
