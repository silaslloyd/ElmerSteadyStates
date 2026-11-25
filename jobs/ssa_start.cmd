#!/bin/bash
#SBATCH --output=./slurms/ssa_start_%j.out
#SBATCH --job-name=ssastart
#SBATCH --ntasks=64
#SBATCH --time=2-00:00:00
#
# A fine grounding-line run from the STEADY SSA PROFILE (grounding line 585.57 km, diag/ssa_profile.py with the model's
# own parameters) instead of Schoof's 686 km start of the pipeline. Everything in one job, every step resumable:
#   1. the SSA profile (diag/ssa_profile.py)            -> work/flowline_gl_fine_ssa/ssa_profile.csv
#   2. the initial-state tables for it (jobs/ssa_tables.py) -> case/marine_ics_ssa
#   3. ess setup + init on configs/flowline_gl_fine_ssa.toml (its own work dir, 250 m band at the SSA grounding line),
#      the shelf break of the pipeline's S0                  -> work/flowline_gl_fine_ssa/start1.npz
#   4. jobs/gl_branch.py from that state (the shelf cut 200 km past the grounding line, the --res band, the Stokes +
#      free-surface transient with the contact grounding line every step, logging "BRANCH <tag>: ...")
#      -> work/flowline_gl_fine_ssa/branch_<TAG>/
#
#   sbatch jobs/ssa_start.cmd
#   RES=125 TAG=ssa125 ARGS="--max-chunks 100" sbatch jobs/ssa_start.cmd
# RES (m, default 125), TAG (default ssa<RES>), ARGS (more options of gl_branch.py; default "--max-chunks 100"). Resubmitting resumes.
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
CFG=configs/flowline_gl_fine_ssa.toml
W=work/flowline_gl_fine_ssa
RES=${RES:-125}
TAG=${TAG:-ssa$RES}
mkdir -p $W
[ -f $W/ssa_profile.csv ] || python3 diag/ssa_profile.py --params case/parameters_fine2x.lua --out $W/ssa_profile
python3 jobs/ssa_tables.py $W/ssa_profile.csv case/marine_ics_ssa
if [ ! -f $W/start1.npz ]; then
    python3 -u -m ess setup $CFG
    python3 -u -m ess init $CFG --out $W/state0.npz
    python3 -u jobs/shelf_break.py $W/state0.npz $W/start1.npz --strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 --width-km 10 --force
fi
python3 - <<PY
import sys
sys.path.insert(0, ".")
from ess.statefile import load_state
from ess.spectrum import gl_position
s = load_state("$W/start1.npz")
print(f"start state: grounding line {gl_position(s.frozen, s.diag) / 1e3:.2f} km")
PY
python3 -u jobs/gl_branch.py --config $CFG --from $W/start1.npz --res $RES --tag $TAG --force ${ARGS:---max-chunks 100}
