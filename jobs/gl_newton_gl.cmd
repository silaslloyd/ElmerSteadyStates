#!/bin/bash
#SBATCH --output=./slurms/gl_newton_gl_%j.out
#SBATCH --job-name=glnewtongl
#SBATCH --ntasks=64
#SBATCH --time=2-00:00:00
#
# Newton with the grounding line as an unknown (jobs/gl_newton_gl.py, ess/glroot.py). The frozen-mask Newton
# (`ess newton --stokes-only`, as jobs/gl_newton.cmd) gives the steady state for the grounding line it is given, and the
# grounding line is almost neutral on this flat bed (a transient needs ~100 kyr), so the steady grounding line is found
# as the root of r(x_g) = x_f - x_g, x_f = the flotation point of the converged state: one Newton per imposed position,
# a first step of 10 km in the direction r points, then a secant method. Starts from a saved state of the branch, so it
# does not touch the running branch.
#
#   TAG=ssa125 WORK=work/gl_ssa sbatch jobs/gl_newton_gl.cmd
#   TAG=ssa125 WORK=work/gl_ssa FROM=work/gl_ssa/branch_ssa125/transient_chunks/points/state_0010.npz sbatch jobs/gl_newton_gl.cmd
# TAG (default ssa125: the branch whose config work/<case>/refined/branch_<TAG>.toml and mesh are used), WORK (default
# work/gl_ssa), FROM (the state to start from; default the newest state of the branch's transient_chunks/points), SLIDING
# (default temperate, as the pipeline's S2), ARGS (more options of jobs/gl_newton_gl.py, e.g. "--step0-km 20 --maxit 25
# --max-evals 10 --r-tol 5"). Output: $WORK/glroot_$TAG/ (steady_KK.npz per evaluation, history.json, RESULT.txt).
# Evaluations already finished are not redone: resubmitting resumes.
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
TAG=${TAG:-ssa125}
W=${WORK:-work/gl_ssa}
CFGB=$W/refined/branch_$TAG.toml
[ -f "$CFGB" ] || { echo "no $CFGB: the branch $TAG has not been set up (jobs/gl_branch.cmd)"; exit 1; }
FROM=${FROM:-$(ls -v $W/branch_$TAG/transient_chunks/points/state_*.npz | tail -1)}
[ -f "$FROM" ] || { echo "no state to start from ($FROM)"; exit 1; }
echo "GL Newton on $TAG: start $FROM, sliding $ESS_SLIDING, out $W/glroot_$TAG"
python3 -u jobs/gl_newton_gl.py --config $CFGB --from "$FROM" --tag $TAG --out-dir $W/glroot_$TAG $ARGS
