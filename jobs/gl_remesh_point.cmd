#!/bin/bash
#SBATCH --output=./slurms/gl_remesh_%j.out
#SBATCH --job-name=glremesh
#SBATCH --ntasks=128
#SBATCH --time=1-00:00:00
#
# A converged sweep point on a mesh with a much finer band (default 20 m) at the head of its temperate patch (jobs/gl_remesh_point.py):
# remesh, transfer the state, solve it again at the same d. Then the stability job on the new state (it prints the command).
#   CFG=work/gl_ssa_gl549/refined/j0000.toml STATE=work/gl_ssa_gl549/sweep_gl549s100r/seg_j0000/points/state_05239.npz sbatch jobs/gl_remesh_point.cmd
# CFG, STATE (required); FINE (m, default 20), UP (km upstream of the head, default 0.3), DOWN (km, default 0.1), GRADE (km, default 1),
# TAG (default R + the head's position); NEWTON_TOL (default 5e-5).
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
[ -n "$CFG" ] && [ -f "$CFG" ] || { echo "CFG not set or no such file: $CFG"; exit 1; }
[ -n "$STATE" ] && [ -f "$STATE" ] || { echo "STATE not set or no such file: $STATE"; exit 1; }
echo "$(date +%T) remesh of $STATE (config $CFG): ${FINE:-20} m band, ${UP:-0.3} km upstream / ${DOWN:-0.1} km downstream of the head"
python3 -u jobs/gl_remesh_point.py "$CFG" "$STATE" --fine ${FINE:-20} --up-km ${UP:-0.3} --down-km ${DOWN:-0.1} --grade-km ${GRADE:-1} \
    $( [ -n "$TAG" ] && echo "--tag $TAG" )
echo "$(date +%T) done"
