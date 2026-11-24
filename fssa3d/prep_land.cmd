#!/bin/bash
#SBATCH --output=./slurms/fssa3d_prep_land_%j.out
#SBATCH --job-name="fssa3d-prep-land"
#SBATCH --nodes=4
#SBATCH --ntasks=256
#SBATCH --time=02:00:00
#
# The 3D set-up of the free-surface / FSSA comparison for a GROUNDED, land-terminating sheet (no shelf, no grounding
# line, no sea pressure) on a strip with free-slip walls. Default config configs/land_flowline_3d.toml: the domain runs
# to L, 2.5 x the initial sheet (its margin at 0.8 L), so the margin is FREE to move (a thin ice-free layer beyond it,
# ablation faded out as the ice thins); configs/land_margin_3d.toml (CFG3=...) is the fixed margin at 0.8 L instead.
#
#   sbatch fssa3d/prep_land.cmd                                      # start: ess init's initial sheet (then spin up)
#   SRC_STATE=work/land_flowline/steady_p268.15_soft.npz sbatch fssa3d/prep_land.cmd   # a 2D state carried onto the strip
#
# Either start is NOT a steady state of the test's physics (Stokes only, temperate sliding, no friction dip): spin it
# up first (jobs/flowline_fssa.cmd, ESS_SLIDING=temperate ESS_SLIDE_YDIP=0, dt 200, tens of kyr), then start the
# sweeps (fssa3d/sweep.sh) from that end state. The mesh is partitioned for this job's ranks: the later jobs must use
# the same --ntasks (default 256 = NODES=4 NT=256). Writes <work>/start_flowline.npz.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28}
export ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install2}
bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
if [ -n "$ELMER_PREFIX" ]; then
    export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
    export ELMER_LIB=$ELMER_PREFIX/share/elmersolver/lib ELMER_SOLVER_HOME=$ELMER_PREFIX/share/elmersolver
    export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/share/elmersolver/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
fi
CFG3=${CFG3:-configs/land_flowline_3d.toml}
WORK=$(python3 -c "import os; from ess.config import load_config as L; print(os.path.relpath(L('$CFG3')['_work']))")
N=${SLURM_NTASKS:-256}
export ESS_NPROCS=$N ESS_MPIRUN="mpirun -np {n} --bind-to none"
python3 -u -m ess setup "$CFG3"
# the target layout (ess init: the initial sheet, and a cheap, inexact Stokes solve whose velocity is not kept; the 3D
# settings of jobs/flowline_fssa.cmd: P1/P1 + bubbles, the cheap block solver)
ESS_STOKES_ELEMENT=${STOKES_ELEMENT:-Bubbles} ESS_LINEAR_STOKES=block ESS_STOKES_BLOCK_FILE=elmer/stokes_cheap.sif.in \
    python3 -u -m ess init "$CFG3" --out "$WORK/state_init.npz"
if [ -n "$SRC_STATE" ]; then
    [ -f "$SRC_STATE" ] || { echo "no $SRC_STATE"; exit 1; }
    python3 -u -m ess transfer "$CFG3" --state "$WORK/state_init.npz" --from "$SRC_STATE" --out "$WORK/start_flowline.npz"
else
    cp "$WORK/state_init.npz" "$WORK/start_flowline.npz"
    echo "no SRC_STATE: the start is ess init's initial sheet"
fi
echo "done: $CFG3, $WORK/start_flowline.npz ($N ranks)"
