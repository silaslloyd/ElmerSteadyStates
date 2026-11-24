#!/bin/bash
#SBATCH --output=./slurms/fssa3d_prep_land_%j.out
#SBATCH --job-name="fssa3d-prep-land"
#SBATCH --nodes=4
#SBATCH --ntasks=256
#SBATCH --time=02:00:00
#
# The 3D set-up of the free-surface / FSSA comparison for a GROUNDED, land-terminating sheet (no shelf, no grounding
# line, no sea pressure): the 2D land steady state carried onto the 3D strip of configs/land_margin_3d.toml (20 km
# wide, free-slip walls, 40 elements across). The friction dip is switched on later (ESS_SLIDE_YDIP, fssa3d/sweep.sh).
#
#   SRC_STATE=work/land_margin/steady_p268.228_flux.npz sbatch fssa3d/prep_land.cmd
#
# SRC_STATE: the converged 2D land state (the one the config's own header uses: README.md, "3D strip"). CFG3 (default
# configs/land_margin_3d.toml): another land 3D config. The mesh is partitioned for this job's ranks: the sweep jobs
# must use the same --ntasks (default 256 = NODES=4 NT=256 of fssa3d/sweep.sh). Writes <work>/start_flowline.npz.
#
#   then:  CFG=configs/land_margin_3d.toml START=work/land_margin_3d/start_flowline.npz \
#          NODES=4 NT=256 DT=25 YEARS=1000 CONTROL=0 VARIANTS="full none adapted" bash fssa3d/sweep.sh
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
: "${SRC_STATE:?SRC_STATE=the converged 2D land state (.npz), e.g. work/land_margin/steady_p268.228_flux.npz}"
[ -f "$SRC_STATE" ] || { echo "no $SRC_STATE"; exit 1; }
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
CFG3=${CFG3:-configs/land_margin_3d.toml}
WORK=$(python3 -c "import os; from ess.config import load_config as L; print(os.path.relpath(L('$CFG3')['_work']))")
N=${SLURM_NTASKS:-256}
export ESS_NPROCS=$N ESS_MPIRUN="mpirun -np {n} --bind-to none"
python3 -u -m ess setup "$CFG3"
# the target layout (ess init: a cheap, inexact Stokes solve whose velocity is not kept; the 3D settings of
# jobs/flowline_fssa.cmd: P1/P1 + bubbles, the cheap block solver), then the 2D state carried onto it
ESS_STOKES_ELEMENT=${STOKES_ELEMENT:-Bubbles} ESS_LINEAR_STOKES=block ESS_STOKES_BLOCK_FILE=elmer/stokes_cheap.sif.in \
    python3 -u -m ess init "$CFG3" --out "$WORK/state_init.npz"
python3 -u -m ess transfer "$CFG3" --state "$WORK/state_init.npz" --from "$SRC_STATE" --out "$WORK/start_flowline.npz"
echo "done: $CFG3, $WORK/start_flowline.npz ($N ranks)"
