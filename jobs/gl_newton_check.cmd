#!/bin/bash
#SBATCH --output=./slurms/gl_newton_check_%j.out
#SBATCH --job-name=glnewtoncheck
#SBATCH --ntasks=64
#SBATCH --time=02:00:00
#
# Does the Newton state stay where it is? One chunk (100 steps x 10 yr = 1000 yr) of the branch's own transient
# (the options of jobs/gl_branch.py: Stokes + free surfaces, contact grounding line every step, temperate bed) from
# the state `ess newton` wrote (jobs/gl_newton.cmd). The grounding line and the mass balance before and after:
# unchanged -> Newton and the transient agree on the steady state; the line retreating as in the branch (about
# 1 km per chunk) -> the two solve different problems.
#
#   sbatch jobs/gl_newton_check.cmd
#   FROM=work/flowline_gl_fine/branch_r125c/newton_state_0010.npz TAG=r125c sbatch jobs/gl_newton_check.cmd
# TAG (default r125c), FROM (default the branch's newton_state_0010.npz), OUT (default next to FROM, newton_check_<name>.npz),
# SLIDING (default temperate, as the branch), CHUNKS (default 1).
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
FROM=${FROM:-$W/branch_$TAG/newton_state_0010.npz}
[ -f "$FROM" ] || { echo "no Newton state ($FROM)"; exit 1; }
OUT=${OUT:-$(dirname $FROM)/newton_check_$(basename $FROM)}
rm -f "$OUT"
cat > ${SLURM_JOB_ID:-local}_gl.py <<'PY'
import sys
sys.path.insert(0, ".")
from ess.statefile import load_state
from ess.spectrum import gl_position
for p in sys.argv[1:]:
    s = load_state(p)
    print(f"  grounding line {gl_position(s.frozen, s.diag) / 1e3:.3f} km  ({p})")
PY
echo "before:"; python3 ${SLURM_JOB_ID:-local}_gl.py $FROM
python3 -u -m ess run $CFGB --state $FROM --stokes-only --gl contact --dt 10 --steps 100 --chunks ${CHUNKS:-1} \
    --save-every 5 --linear-stokes direct --out $OUT
echo "after:"; python3 ${SLURM_JOB_ID:-local}_gl.py $OUT
rm -f ${SLURM_JOB_ID:-local}_gl.py
