#!/bin/bash
#SBATCH --output=./slurms/gl_sweep_%j.out
#SBATCH --job-name=glsweep
#SBATCH --ntasks=128
#SBATCH --time=3-00:00:00
#
# The temperate-patch sweep from the thermomechanical steady state of a grounding-line branch (jobs/gl_newton_thermal.cmd:
# steady_temperate.npz, the whole grounded bed temperate): the patch length d from the grounding line upstream goes down in
# STEP km steps (default 50) to D_END (default 0), Newton at each d with the masks frozen and the grounding line moved by contact
# between solves (jobs/sweep_refined.py, `ess sweep`). One node, 128 tasks (every segment's mesh is partitioned for them).
#
# MESH. The base config is the branch's with the 125 m grounding-line band narrowed to BELOW / ABOVE km (default 35 / 25: half of
# the branch's 70 / 50) around the grounding line of the state, graded to 2 km over GRADE km (default 30). The temperate head has
# a FINE m band (default 100) over the heads a segment covers: SPAN km of head range (default 100 = TWO steps of 50) + PAD km
# either side (default 10), graded over 10 km. Each segment on its own mesh, set up and transferred from the last converged point.
# SHORT HORIZON: Newton's map is MAP_STEPS steps of the config's dt (default 10 x 50 yr = 500 yr; the branch's Newton had 40 x 50
# = 2000 yr). The steady states are the same, an unstable mode's growth over the map is not. MAP_STEPS=0: the config's own.
#
#   GL=549 sbatch jobs/gl_sweep.cmd
#   GL=549 D_END=300 sbatch jobs/gl_sweep.cmd                 # only part of the way
#   GL=549 BELOW=70 ABOVE=50 FINE=125 SPAN=50 STEP=25 sbatch jobs/gl_sweep.cmd
# GL (default 549), SHELF (default 100), TAG (default gl<GL>s<SHELF>), STATE (default work/gl_ssa_gl<GL>/branch_<TAG>/steady_temperate.npz),
# D_START (km, default the grounding line's position: the whole grounded bed), D_END, STEP, SPAN, PAD, FINE, BELOW, ABOVE, GRADE,
# MAP_STEPS, PREDICTOR (none|secant), NEWTON_TOL (default 5e-5), ITS_MAX (default 8: a Newton with more iterations halves the next
# steps, they double back once a solve takes <= ITS_MAX / 2; a failure halves the step too; 0: off), FLOOR (km, default 1: the smallest
# step), BISECT (km, default 2: where the consistency of the steady state changes between two points -- cold bed above melting, or
# the patch's water sheet (CS hydrology on its melt) running dry inside it -- the interval is bisected down to this length; the midpoints in sweep_<TAG>/seg_*/bisect.csv
# and points/; 0: off). Output: work/gl_ssa_gl<GL>/sweep_<TAG>/seg_mXXXX/ (sweep.csv, points/),
# segment configs work/gl_ssa_gl<GL>/refined/mXXXX.toml. Resubmitting resumes (finished segments are skipped).
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
export ESS_NEWTON_TOL=${NEWTON_TOL:-5e-5}
GL=${GL:-549}
SHELF=${SHELF:-100}
TAG=${TAG:-gl${GL}s${SHELF}}
W=work/gl_ssa_gl$GL
BCFG=$W/refined/branch_$TAG.toml
[ -f "$BCFG" ] || { echo "no $BCFG: the branch $TAG has not been set up"; exit 1; }
STATE=${STATE:-$W/branch_$TAG/steady_temperate.npz}
[ -f "$STATE" ] || { echo "no $STATE: run jobs/gl_newton_thermal.cmd first"; exit 1; }
BELOW=${BELOW:-35}
ABOVE=${ABOVE:-25}
GRADE=${GRADE:-30}
STEP=${STEP:-50}
SPAN=${SPAN:-100}
PAD=${PAD:-10}
FINE=${FINE:-100}
MAP_STEPS=${MAP_STEPS:-10}
CFG=$W/refined/sweep_$TAG.toml
OUTROOT=sweep_$TAG
# the base config: the branch's with the grounding-line band narrowed; prints the grounding line (the whole grounded bed = d)
GLKM=$(python3 - <<PY
import re, sys
sys.path.insert(0, "."); sys.path.insert(0, "jobs")
from ess.statefile import load_state
from ess.spectrum import gl_position
from sweep_refined import banded_points, write_config
s = load_state("$STATE")
xgl = gl_position(s.frozen, s.diag) / 1e3
text = open("$BCFG").read()
front = float(re.search(r"^front_km\s*=\s*([\d.]+)", text, flags=re.M).group(1))
pts = banded_points([[0.0, 2000.0], [front, 2000.0]], xgl - $BELOW, xgl + $ABOVE, 125.0, $GRADE, front)
write_config("$BCFG", "$CFG", pts, "{work}/branchcase_$TAG")
print(f"{xgl:.4f}")
PY
)
DSTART=${D_START:-$GLKM}
echo "$(date +%T) sweep on $TAG from $STATE: grounding line $GLKM km, d $DSTART -> ${D_END:-0} km in ${STEP} km steps; 125 m band $BELOW km below / $ABOVE km above the grounding line (graded $GRADE km); head band ${FINE} m, segments of $SPAN km (+ $PAD km); map ${MAP_STEPS} steps; steps halved above ${ITS_MAX:-8} Newton its (floor ${FLOOR:-1} km); bisect to ${BISECT:-2} km; 128 tasks"
echo "  base config $CFG"
python3 -u jobs/sweep_refined.py $CFG --from-state $STATE --d-km $DSTART --d-end-km ${D_END:-0} \
    --span-km $SPAN --pad-km $PAD --fine $FINE --grade-km 10 --step-frac 1 --step-min-km $STEP --step-max-km $STEP \
    --predictor ${PREDICTOR:-none} --newton-its-max ${ITS_MAX:-8} --step-floor-km ${FLOOR:-1} --bisect-km ${BISECT:-2} --out-root $OUTROOT --tag-prefix s \
    $( [ "$MAP_STEPS" -gt 0 ] && echo "--map-steps $MAP_STEPS" )
echo "$(date +%T) done"
