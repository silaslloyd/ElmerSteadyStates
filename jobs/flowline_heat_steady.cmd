#!/bin/bash
#SBATCH --output=./slurms/flowline_heat_%j.out
#SBATCH --job-name=flheat
#SBATCH --ntasks=128
#SBATCH --time=1-00:00:00
#
# The steady temperature on a fixed flowline geometry (configs/flowline_rect_ext.toml), thermomechanically coupled
# through the bed: subtemperate sliding (the config's), frictional heating (flux C(T) |u_b|^2) and strain heating,
# Stokes and heat solved on the geometry of STATE (no free surface: ess run --heat-only), the grounded mask fixed as in
# STATE and the cold/temperate mask fixed TEMPERATE on the whole bed (ess run --bed temperate: ColdTempMask +1, the
# bed at T0, Dirichlet; --no-remask). Each chunk: 2 steps of 1e6 yr (the steady temperature for that velocity); chunks
# until the rms change is below TOL (default 1e-4 scaled: 0.5 mK) or CHUNKS (default 20).
# STATE (default): the converged mechanics-only temperate Newton state at the contact grounding line, from the run
# cache (jobs/state_from_cache.py --at AT, default the last evaluation of job 31865's first solve, 2026-11-18 00:14:16)
#   -> work/flowline_rect_ext/bound_temperate_newton0.npz
# Output: work/flowline_rect_ext/thermo_heat.npz (and every chunk in thermo_heat_chunks/points/).
# Variants (env, passed through by sbatch): ESS_STRAIN_HEAT_POSITIVE=1 (the strain heating only where positive: the
# nodal projection overshoots below zero at the grounding line); ESS_NO_CONVECTION=1 (no advection, a comparison);
# ESS_HEAT_LIMITER=1 (the ice capped at melting, T <= T0: the excess heat goes to melt).
#   sbatch jobs/flowline_heat_steady.cmd
#   ESS_STRAIN_HEAT_POSITIVE=1 OUT=work/flowline_rect_ext/thermo_heat_pos.npz sbatch jobs/flowline_heat_steady.cmd
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
# the sea-pressure spring (rho_w g dt at the floating base) at the map's dt, not the 1e6 yr of these steps: with the
# step's dt it is 20000x stiffer, pins the floating base and stalls the ice next to the grounding line (u ~ 0 in a
# wedge after it, 2026-11-18), so the velocity would not be the steady state's
export ESS_SEA_SPRING_DT=${ESS_SEA_SPRING_DT:-50}
CFG=configs/flowline_rect_ext.toml
W=work/flowline_rect_ext
if [ -z "$STATE" ]; then
    STATE=$W/bound_temperate_newton0.npz
    [ -f $STATE ] || python3 jobs/state_from_cache.py $W $W/bound_temperate_now.npz $STATE --at "${AT:-2026-11-18 00:14:16}"
fi
OUT=${OUT:-$W/thermo_heat.npz}
echo "$(date +%T) steady heat (subtemperate sliding, friction + strain heating) on the geometry of $STATE -> $OUT"
[ "$ESS_STRAIN_HEAT_POSITIVE" = 1 ] && echo "  strain heating: positive part only"
[ "$ESS_NO_CONVECTION" = 1 ] && echo "  NO advection in the ice (comparison)"
[ "$ESS_HEAT_LIMITER" = 1 ] && echo "  heat limiter: T <= T0 in the ice"
echo "  sea spring time step $ESS_SEA_SPRING_DT yr"
python3 -c "
import sys, numpy as np; sys.path.insert(0, '.')
from ess.statefile import load_state, find_name
s = load_state('$STATE')
for n in ('GroundedMask', 'ColdTempMask'):
    v = s.frozen[find_name(s.frozen, n)][1]
    print(f'  {n} (in the state; ColdTempMask then set +1 everywhere): {int((v > 0.5).sum())} of {len(v)} nodes +1, {int((np.abs(v) < 0.5).sum())} 0, {int((v < -0.5).sum())} -1')"
python3 -u -m ess run $CFG --state $STATE --heat-only --gl frozen --bed temperate --no-remask --chunks ${CHUNKS:-20} --steps 2 \
    --dt 1e6 --until-tol ${TOL:-1e-4} --save-every 1 --out $OUT
echo "$(date +%T) done"
