#!/bin/bash
#SBATCH --output=./slurms/fssa3d_prep_%j.out
#SBATCH --job-name="fssa3d-prep"
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=02:00:00
#
# The 3D set-up for the free-surface / FSSA comparison: the converged flowline steady state (temperate, Stokes) extruded
# across a strip with free-slip walls; the friction dip is switched on later (ESS_SLIDE_YDIP, fssa3d/sweep.sh).
#
#   SRC_CFG=work/flowline_gl_fine/refined/pipe1.toml SRC_STATE=<the converged flowline state .npz> \
#       sbatch fssa3d/prep.cmd
#
# Writes configs/flowline_gl_3d.toml (fssa3d/make_config.py: WIDTH 20 km, Y_ELEMENTS 20, MIN_RES 500 m), builds the mesh
# for this job's ranks (the sweep jobs must use the same --ntasks) and work/flowline_gl_3d/start_flowline.npz.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
: "${SRC_CFG:?SRC_CFG=the flowline config (e.g. work/flowline_gl_fine/refined/pipe1.toml)}"
: "${SRC_STATE:?SRC_STATE=the converged flowline state (.npz)}"
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
CFG3=${CFG3:-configs/flowline_gl_3d.toml}
python3 fssa3d/make_config.py "$SRC_CFG" "$CFG3" --width ${WIDTH:-20000} --y-elements ${Y_ELEMENTS:-20} \
    --min-res ${MIN_RES:-500} --dip 0.0 --dip-width ${DIP_WIDTH:-2} --work ${WORK3:-work/flowline_gl_3d}
WORK=$(python3 -c "import os; from ess.config import load_config as L; print(os.path.relpath(L('$CFG3')['_work']))")
N=${SLURM_NTASKS:-64}
export ESS_NPROCS=$N ESS_MPIRUN="mpirun -np {n} --bind-to none"
python3 -u -m ess setup "$CFG3"
# the target layout (ess init: a cheap, inexact Stokes solve whose velocity is not kept; the 3D settings of
# jobs/flowline_fssa.cmd: P1/P1 + bubbles, the cheap block solver), then the flowline state carried onto it
ESS_STOKES_ELEMENT=${STOKES_ELEMENT:-Bubbles} ESS_LINEAR_STOKES=block ESS_STOKES_BLOCK_FILE=elmer/stokes_cheap.sif.in \
    python3 -u -m ess init "$CFG3" --out "$WORK/state_init.npz"
python3 -u -m ess transfer "$CFG3" --state "$WORK/state_init.npz" --from "$SRC_STATE" --out "$WORK/start_flowline.npz"
echo "done: $CFG3, $WORK/start_flowline.npz ($N ranks)"
