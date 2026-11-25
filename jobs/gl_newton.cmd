#!/bin/bash
#SBATCH --output=./slurms/gl_newton_%j.out
#SBATCH --job-name=glnewton
#SBATCH --ntasks=64
#SBATCH --time=2-00:00:00
#
# The Newton version of a grounding-line branch (jobs/gl_branch.py): instead of time stepping the 125 m run for
# days, Newton on the steady state (`ess newton --stokes-only`, as the pipeline's S2): the geometry (Zs, Zb) that is
# steady, the grounding line moved by the contact criterion between converged solves ([masks] gl = "contact") and
# solved again, until the contact criterion agrees with the grounding line. It starts from a saved state of the branch,
# so it does not touch the running branch.
#
#   TAG=r125c sbatch jobs/gl_newton.cmd
#   TAG=r125c FROM=work/flowline_gl_fine/branch_r125c/transient_chunks/points/state_0010.npz sbatch jobs/gl_newton.cmd
# TAG (default r125c: the branch whose config work/<case>/refined/branch_<TAG>.toml and mesh are used), FROM (the state
# to start from; default the newest state of the branch's transient_chunks/points), OUT (default
# work/<case>/branch_<TAG>/newton_<FROM's name>.npz), SLIDING (default temperate, as the pipeline's S2; subtemperate:
# the config's own sliding, as the branch's transient), ARGS (more options of `ess newton`, e.g. "--maxit 25").
# The Newton iterates are checkpointed (newton_checkpoint.npz in the work directory): resubmitting resumes.
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
unset ESS_HEAT_LIMITER ESS_SEA_SPRING_DT ESS_STRAIN_HEAT_POSITIVE ESS_NO_CONVECTION
export ESS_SLIDING=${SLIDING:-temperate} ESS_CONTACT_FUNCTION=elmerstreams
TAG=${TAG:-r125c}
W=${WORK:-work/flowline_gl_fine}
CFGB=$W/refined/branch_$TAG.toml
[ -f "$CFGB" ] || { echo "no $CFGB: the branch $TAG has not been set up (jobs/gl_branch.cmd)"; exit 1; }
FROM=${FROM:-$(ls -v $W/branch_$TAG/transient_chunks/points/state_*.npz | tail -1)}
[ -f "$FROM" ] || { echo "no state to start from ($FROM)"; exit 1; }
OUT=${OUT:-$W/branch_$TAG/newton_$(basename $FROM .npz).npz}
START=$W/branch_$TAG/newton_start_$(basename $FROM .npz).npz
cp -n "$FROM" "$START"             # a copy: the branch may be running and rewrite its files
echo "Newton on $TAG: start $FROM, sliding $ESS_SLIDING, out $OUT"
python3 -u -m ess newton $CFGB --state $START --stokes-only --out $OUT $ARGS
echo "grounding line of the result:"
python3 - <<PY
import sys
sys.path.insert(0, ".")
from ess.statefile import load_state
from ess.spectrum import gl_position
s = load_state("$OUT")
print(f"  {gl_position(s.frozen, s.diag) / 1e3:.2f} km")
PY
