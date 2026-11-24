#!/bin/bash
#SBATCH --output=./slurms/gl_hold_%j.out
#SBATCH --job-name=glhold
#SBATCH --ntasks=64
#SBATCH --time=06:00:00
#
# Does the pipeline's converged state STAY where it is on the mesh it ended on? The S1 transient (Stokes + free surface,
# temperate bed, contact grounding line every step, dt 10 yr, 100 steps = 1000 yr per chunk) for a few chunks, from the
# S2 state, on the pipeline's own refined mesh (no remeshing, no transfer), logging after each chunk:
#   HOLD: chunk k (yr): grounding line G km (change since the start), rms change
# The same transient as the 125 m branch (jobs/gl_branch.py), so the two read side by side.
#   sbatch jobs/gl_hold.cmd
# STATE (default work/flowline_gl_fine/pipeline/stokes_newton.npz, the S2 state), CONFIG (default
# work/flowline_gl_fine/refined/pipe1.toml, the pipeline's remeshed config), CHUNKS (default 3), TAG (default hold250):
# states in work/flowline_gl_fine/<TAG>/ (transient.npz, and every chunk's in transient_chunks/points/).
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
export ESS_CONTACT_FUNCTION=elmerstreams ESS_SLIDING=temperate

CONFIG=${CONFIG:-work/flowline_gl_fine/refined/pipe1.toml}
STATE=${STATE:-work/flowline_gl_fine/pipeline/stokes_newton.npz}
TAG=${TAG:-hold250}
CHUNKS=${CHUNKS:-3}
[ -f "$STATE" ] || { echo "no $STATE"; exit 1; }
[ -f "$CONFIG" ] || { echo "no $CONFIG"; exit 1; }
OUTDIR=$(dirname "$(dirname "$STATE")")/$TAG       # work/flowline_gl_fine/<TAG>
mkdir -p "$OUTDIR"
OUT=$OUTDIR/transient.npz

gl() {   # grounding line (km) of a state file
    python3 -c "
import sys; sys.path.insert(0, '.')
from ess.statefile import load_state; from ess.spectrum import gl_position
s = load_state('$1'); print(f'{gl_position(s.frozen, s.diag) / 1e3:.3f}')"
}
G0=$(gl "$STATE")
echo "$(date +%T) HOLD: start state $STATE on $CONFIG: grounding line $G0 km"
CUR=$STATE
for ((k = 0; k < CHUNKS; k++)); do
    python3 -u -m ess run "$CONFIG" --state "$CUR" --stokes-only --gl contact --dt 10 --steps 100 --chunks 1 \
        --save-every 1 --chunk-offset $k --linear-stokes direct --out "$OUT" 2>&1 | tee "$OUTDIR/chunk_$k.log" \
        | grep -E "rms change|contact grounding line" | tail -n 2 | sed 's/^/    /'
    CUR=$OUT
    G=$(gl "$OUT")
    RMS=$(grep -h "rms change" "$OUTDIR/chunk_$k.log" | tail -1 | sed 's/.*rms change *//')
    python3 -c "print(f'$(date +%T) HOLD: chunk {$k + 1} ({($k + 1) * 1000} yr): grounding line $G km ({$G - $G0:+.3f} since the start), rms change $RMS')"
done
echo "$(date +%T) HOLD: done, states in $OUTDIR"
