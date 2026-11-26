#!/bin/bash
#SBATCH --output=./slurms/gl_newton_thermal_%j.out
#SBATCH --job-name=glnewtherm
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# The thermomechanical steady state of a grounding-line branch by Newton: from the steady temperature of jobs/gl_heat_steady.cmd
# (thermo_heat.npz: fixed geometry, friction + strain heating, the whole bed temperate), `ess newton` on the full map
# (temperature, Zs, Zb; Stokes + free surface + heat, the config's subtemperate sliding, friction and strain heating, no heat
# limiter, the sea spring at the map's dt), with the MASKS FIXED: the grounded mask (the grounding line) and the cold/temperate
# mask (the bed temperate) stay as in STATE ([masks] auto = false in a copy of the branch's config, as jobs/gl_newton_gl.py): the
# polish of the full state for the grounding line it has, no remask rounds by the contact criterion. The Newton iterates are
# checkpointed (newton_checkpoint.npz in the work directory): resubmitting resumes. Do not run two Newtons of one work
# directory at once.
#
#   GL=549 sbatch jobs/gl_newton_thermal.cmd
#   GL=549 TAG=gl549s100 STATE=work/gl_ssa_gl549/branch_gl549s100/thermo_heat.npz ARGS="--maxit 25" sbatch jobs/gl_newton_thermal.cmd
# GL (km of the start, default 549), SHELF (km, default 100), TAG (default gl<GL>s<SHELF>: the branch whose config
# work/gl_ssa_gl<GL>/refined/branch_<TAG>.toml and mesh are used), STATE (default work/gl_ssa_gl<GL>/branch_<TAG>/thermo_heat.npz),
# OUT (default .../branch_<TAG>/steady_temperate.npz), ARGS (more options of `ess newton`).
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
unset ESS_SLIDING ESS_HEAT_LIMITER ESS_SEA_SPRING_DT ESS_STRAIN_HEAT_POSITIVE ESS_NO_CONVECTION   # the config's subtemperate sliding
export ESS_CONTACT_FUNCTION=elmerstreams
GL=${GL:-549}
SHELF=${SHELF:-100}
TAG=${TAG:-gl${GL}s${SHELF}}
W=work/gl_ssa_gl$GL
CFG=$W/refined/branch_$TAG.toml
[ -f "$CFG" ] || { echo "no $CFG: the branch $TAG has not been set up (jobs/gl_cut_shelf.cmd, jobs/ssa_start_gl.cmd)"; exit 1; }
STATE=${STATE:-$W/branch_$TAG/thermo_heat.npz}
[ -f "$STATE" ] || { echo "no $STATE: run jobs/gl_heat_steady.cmd first"; exit 1; }
OUT=${OUT:-$W/branch_$TAG/steady_temperate.npz}
CFGN=$W/branch_$TAG/newton_fixed.toml
python3 - <<PY
import sys
sys.path.insert(0, "jobs")
from gl_newton_gl import set_key
open("$CFGN", "w").write(set_key(open("$CFG").read(), "masks", "auto", "false"))
PY
grep -n -A6 "^\[masks\]" $CFGN | head -8
echo "$(date +%T) thermomechanical Newton on $TAG: start $STATE (bed temperate, subtemperate sliding, masks fixed) -> $OUT"
python3 -u -m ess newton $CFGN --state $STATE --out $OUT $ARGS
echo "grounding line of the result:"
python3 - <<PY
import sys
sys.path.insert(0, ".")
from ess.statefile import load_state
from ess.spectrum import gl_position
s = load_state("$OUT")
print(f"  {gl_position(s.frozen, s.diag) / 1e3:.2f} km")
PY
echo "$(date +%T) done"
