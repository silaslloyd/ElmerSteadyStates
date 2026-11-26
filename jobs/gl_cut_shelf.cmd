#!/bin/bash
#SBATCH --output=./slurms/gl_cut_shelf_%j.out
#SBATCH --job-name=glcut
#SBATCH --ntasks=64
#SBATCH --time=0-04:00:00
#
# A state of a grounding-line branch (jobs/ssa_start_gl.cmd, jobs/gl_branch.py) on a NEW mesh with a shorter ice shelf: the
# shelf cut SHELF km past the grounding line (default 100; the branches' default is 200), the 125 m band around the grounding
# line as the branch's. jobs/gl_branch.py --max-chunks 0: ess setup / init / transfer / shelf_break, no transient. The new
# branch is work/gl_ssa_gl<GL>/branch_<TAG>/ (start.npz = the cut state, its config work/gl_ssa_gl<GL>/refined/branch_<TAG>.toml),
# TAG default gl<GL>s<SHELF>. The running branches are not touched (own mesh dir branchcase_<TAG>).
#
#   GL=549 SHELF=100 sbatch jobs/gl_cut_shelf.cmd
#   GL=549 SHELF=100 FROM=work/gl_ssa_gl549/branch_gl549/transient_chunks/points/state_0015.npz sbatch jobs/gl_cut_shelf.cmd
# Then e.g. the steady temperature on it:
#   GL=549 TAG=gl549s100 FROM=work/gl_ssa_gl549/branch_gl549s100/start.npz sbatch jobs/gl_heat_steady.cmd
# GL (km of the start, default 549), SHELF (km, default 100), TAG, FROM (default the newest state of branch gl<GL>), RES (m, default 125).
# 64 tasks: the mesh is partitioned for the job's tasks, as the other branch jobs'.
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
unset ESS_SLIDING ESS_HEAT_LIMITER ESS_SEA_SPRING_DT ESS_STRAIN_HEAT_POSITIVE ESS_NO_CONVECTION
export ESS_CONTACT_FUNCTION=elmerstreams
GL=${GL:-549}
SHELF=${SHELF:-100}
RES=${RES:-125}
TAG=${TAG:-gl${GL}s${SHELF}}
W=work/gl_ssa_gl$GL
CFG=configs/flowline_gl_fine_ssa_gl$GL.toml
python3 jobs/ssa_start_gl_cfg.py --gl-km $GL --out $CFG
FROM=${FROM:-$(ls -v $W/branch_gl$GL/transient_chunks/points/state_*.npz | tail -1)}
[ -f "$FROM" ] || { echo "no state to start from ($FROM)"; exit 1; }
echo "$(date +%T) shelf cut to $SHELF km, ${RES} m band, from $FROM -> $W/branch_$TAG"
python3 -u jobs/gl_branch.py --config $CFG --from $FROM --res $RES --tag $TAG --cut-shelf-km $SHELF --force --max-chunks 0
echo "$(date +%T) done: $W/branch_$TAG/start.npz"
