#!/bin/bash
#SBATCH --output=./slurms/flowline_heat_%j.out
#SBATCH --job-name=flheat
#SBATCH --ntasks=128
#SBATCH --time=1-00:00:00
#
# The steady temperature on a fixed flowline geometry (configs/flowline_rect_ext.toml), thermomechanically coupled
# through the bed: subtemperate sliding (the config's), frictional heating (flux C(T) |u_b|^2) and strain heating,
# Stokes and heat solved on the geometry of STATE (no free surface: ess run --heat-only), the grounded mask kept as it
# is (--keep-gl), the cold/temperate mask moved by melting / refreezing between chunks. Each chunk: 2 steps of 1e6 yr
# (the steady temperature for that velocity and mask); chunks until the rms change is below TOL (default 1e-4 scaled:
# 0.5 mK) or CHUNKS (default 20).
# STATE (default): the converged mechanics-only temperate Newton state at the contact grounding line, from the run
# cache (jobs/state_from_cache.py --at AT, default the last evaluation of job 31865's first solve, 2026-11-18 00:14)
#   -> work/flowline_rect_ext/bound_temperate_newton0.npz
# Output: work/flowline_rect_ext/thermo_heat.npz (and every chunk in thermo_heat_chunks/points/).
#   sbatch jobs/flowline_heat_steady.cmd
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
unset ESS_SLIDING                              # the config's subtemperate sliding
CFG=configs/flowline_rect_ext.toml
W=work/flowline_rect_ext
if [ -z "$STATE" ]; then
    STATE=$W/bound_temperate_newton0.npz
    [ -f $STATE ] || python3 jobs/state_from_cache.py $W $W/bound_temperate_now.npz $STATE --at "${AT:-2026-11-18 00:14}"
fi
OUT=${OUT:-$W/thermo_heat.npz}
echo "$(date +%T) steady heat (subtemperate sliding, friction + strain heating) on the geometry of $STATE -> $OUT"
python3 -u -m ess run $CFG --state $STATE --heat-only --gl frozen --keep-gl --chunks ${CHUNKS:-20} --steps 2 \
    --dt 1e6 --until-tol ${TOL:-1e-4} --save-every 1 --out $OUT
echo "$(date +%T) done"
