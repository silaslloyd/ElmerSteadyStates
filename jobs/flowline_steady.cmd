#!/bin/bash
#SBATCH --output=./slurms/flowline_steady_%j.out
#SBATCH --job-name=flsteady
#SBATCH --ntasks=32
#SBATCH --time=3-00:00:00
#
# A 2D steady state of the extended strip's flowline (configs/flowline_rect_ext.toml) by Newton-Krylov, as the starting
# point of the 3D runs. Steps, each skipped when its output exists (resubmitting resumes; finished Elmer runs are cached):
#   1. setup: the flowline mesh (1116 km, 500 m, 1 km wide), ess init, the cold Marine_Term flowline state carried onto
#      it (ess transfer: the bed continued linearly beyond the old 816 km)              -> work/flowline_rect_ext/start.npz
#   2. the shelf break, as the 3D strip's (jobs/shelf_break.py --strip-divide-r-km 1520.8 --r-break-km 450 --factor 30
#      --width-km 10)                                                                    -> bed30.npz
#   3. a spin-up by time stepping: contact grounding line, masks moved between chunks, the bedrock's heat capacity
#      x ROCK_CAP (default 0.001: only the steady state matters), SPIN_CHUNKS (default 20) x SPIN_STEPS (100) x SPIN_DT
#      (10 yr)                                                                           -> spin.npz
#   4. Newton-Krylov from it ([map] of the config; masks frozen per solve, moved between solves: [masks] auto)
#                                                                                         -> steady_p268.15.npz
# Run from the repository root:
#   sbatch jobs/flowline_steady.cmd
# The Elmer of the production runs (ELMER_MODULE / ELMER_PREFIX as jobs/scaling_sweep.sh): the libraries are then the
# same build and are not rebuilt under a running production job.
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
CFG=configs/flowline_rect_ext.toml
W=work/flowline_rect_ext
FROM=${FROM:-work/marine_flowline/run_cold_p268.15.npz}
echo "$(date +%T) Elmer $(command -v ElmerSolver), $SLURM_NTASKS ranks, work dir $W"
mkdir -p $W
if [ ! -f $W/start.npz ]; then
    echo "$(date +%T) 1. setup, init, transfer from $FROM"
    python3 -u -m ess setup $CFG
    python3 -u -m ess init $CFG --out $W/state0.npz
    python3 -u -m ess transfer $CFG --state $W/state0.npz --from $FROM --out $W/start.npz
fi
if [ ! -f $W/bed30.npz ]; then
    echo "$(date +%T) 2. the shelf break"
    python3 jobs/shelf_break.py $W/start.npz $W/bed30.npz --strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 \
        --width-km 10 --force
fi
if [ ! -f $W/spin.npz ]; then
    echo "$(date +%T) 3. spin-up: ${SPIN_CHUNKS:-20} x ${SPIN_STEPS:-100} x ${SPIN_DT:-10} yr, rock heat capacity x ${ROCK_CAP:-0.001}"
    python3 -u -m ess run $CFG --state $W/bed30.npz --chunks ${SPIN_CHUNKS:-20} --steps ${SPIN_STEPS:-100} \
        --dt ${SPIN_DT:-10} --rock-cap ${ROCK_CAP:-0.001} --save-every 5 --out $W/spin.npz
fi
echo "$(date +%T) 4. Newton-Krylov from the spin-up"
python3 -u -m ess newton $CFG --state $W/spin.npz --out $W/steady_p268.15.npz
echo "$(date +%T) done: $W/steady_p268.15.npz"
