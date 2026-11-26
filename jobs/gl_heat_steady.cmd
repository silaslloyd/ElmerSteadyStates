#!/bin/bash
#SBATCH --output=./slurms/gl_heat_steady_%j.out
#SBATCH --job-name=glheat
#SBATCH --ntasks=64
#SBATCH --time=1-00:00:00
#
# The steady temperature on the fixed geometry of a grounding-line branch (jobs/ssa_start_gl.cmd, jobs/gl_branch.py): the
# recipe of jobs/flowline_heat_steady.cmd on the branch's own config and mesh. Stokes and heat are solved on the geometry of
# the state (no free surface: ess run --heat-only), the grounded mask fixed as in the state (--gl frozen), the cold/temperate
# mask fixed TEMPERATE on the whole bed (--bed temperate: ColdTempMask +1, the bed at T0; --no-remask), with the config's
# subtemperate sliding, frictional heating (flux C(T) |u_b|^2) and strain heating. Each chunk: 2 steps of 1e6 yr (the steady
# temperature for that velocity); chunks until the rms change is below TOL (default 1e-4 scaled: 0.5 mK) or CHUNKS (default 20).
# The branch's own mesh is used as it is (125 m band at the grounding line, 2 km in the interior): no remeshing.
#
#   GL=549 sbatch jobs/gl_heat_steady.cmd
#   GL=549 FROM=work/gl_ssa_gl549/branch_gl549/transient_chunks/points/state_0010.npz sbatch jobs/gl_heat_steady.cmd
# GL (km of the start, default 549: the work dir work/gl_ssa_gl<GL>), TAG (default gl<GL>), FROM (the state; default the newest state of
# the branch's transient_chunks/points), ARGS (more options of `ess run`, e.g. "--p 267.15": another Tsurf than the state's), OUT (default work/gl_ssa_gl<GL>/branch_<TAG>/thermo_heat.npz; every chunk in
# thermo_heat_chunks/points/), TOL, CHUNKS. Variants (env, passed through): ESS_STRAIN_HEAT_POSITIVE=1 (strain heating only where
# positive), ESS_NO_CONVECTION=1, ESS_HEAT_LIMITER=1 (ice capped at melting, T <= T0).
# Does not touch the running branch (reads one of its states).
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
export ESS_CONTACT_FUNCTION=elmerstreams
# the sea-pressure spring at the map's dt, not the 1e6 yr of these steps (jobs/flowline_heat_steady.cmd: with the step's dt it
# is 20000x stiffer and pins the floating base next to the grounding line)
export ESS_SEA_SPRING_DT=${ESS_SEA_SPRING_DT:-50}
GL=${GL:-549}
TAG=${TAG:-gl$GL}
W=work/gl_ssa_gl$GL
CFG=$W/refined/branch_$TAG.toml
[ -f "$CFG" ] || { echo "no $CFG: the branch $TAG has not been set up (jobs/ssa_start_gl.cmd)"; exit 1; }
FROM=${FROM:-$(ls -v $W/branch_$TAG/transient_chunks/points/state_*.npz | tail -1)}
[ -f "$FROM" ] || { echo "no state to start from ($FROM)"; exit 1; }
OUT=${OUT:-$W/branch_$TAG/thermo_heat.npz}
echo "$(date +%T) steady heat (subtemperate sliding, friction + strain heating) on the geometry of $FROM -> $OUT"
[ "$ESS_STRAIN_HEAT_POSITIVE" = 1 ] && echo "  strain heating: positive part only"
[ "$ESS_NO_CONVECTION" = 1 ] && echo "  NO advection in the ice (comparison)"
[ "$ESS_HEAT_LIMITER" = 1 ] && echo "  heat limiter: T <= T0 in the ice"
echo "  sea spring time step $ESS_SEA_SPRING_DT yr; config $CFG"
python3 - <<PY
import sys
import numpy as np
sys.path.insert(0, ".")
from ess.statefile import load_state, find_name
from ess.spectrum import gl_position
s = load_state("$FROM")
print(f"  grounding line of the state: {gl_position(s.frozen, s.diag) / 1e3:.2f} km")
for n in ("GroundedMask", "ColdTempMask"):
    v = s.frozen[find_name(s.frozen, n)][1]
    print(f"  {n} (in the state; ColdTempMask then set +1 everywhere): {int((v > 0.5).sum())} of {len(v)} nodes +1, {int((np.abs(v) < 0.5).sum())} 0, {int((v < -0.5).sum())} -1")
PY
python3 -u -m ess run $CFG --state $FROM --heat-only --gl frozen --bed temperate --no-remask --chunks ${CHUNKS:-20} --steps 2 \
    --dt 1e6 --until-tol ${TOL:-1e-4} --save-every 1 --out $OUT $ARGS
echo "$(date +%T) done"
