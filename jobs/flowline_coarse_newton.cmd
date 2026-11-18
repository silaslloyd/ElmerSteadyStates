#!/bin/bash
#SBATCH --output=./slurms/flowline_coarse_%j.out
#SBATCH --job-name=flcoarse
#SBATCH --ntasks=64
#SBATCH --time=3-00:00:00
#
# The thermomechanical steady state with a fixed, fully temperate bed on the coarse flowline mesh
# (configs/flowline_rect_coarse.toml). Steps, each skipped when its output exists:
#   1. setup (mesh, partitioned for the job's ranks), ess init, the fine flowline's thermal steady state FROM
#      (default work/flowline_rect_ext/thermo_heat_lim50.npz) carried onto it (ess transfer)        -> start0.npz
#   2. the shelf break, as jobs/flowline_steady.cmd (the bed of the fine flowline)                  -> start1.npz
#   3. the whole bed temperate (ColdTempMask +1 everywhere; the grounded mask as transferred)       -> start.npz
#   4. Newton (temperature, Zs, Zb; no heat limiter; the sea spring at the map's dt), then rounds of the contact
#      grounding line ([masks] gl = "contact": a short contact run from the converged state; if its grounded mask
#      differs, Newton again with it), the bed kept temperate ([masks] ct = false)              -> steady_temperate.npz
#   sbatch jobs/flowline_coarse_newton.cmd
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
CFG=configs/flowline_rect_coarse.toml
W=work/flowline_rect_coarse
FROM=${FROM:-work/flowline_rect_ext/thermo_heat_lim50.npz}
mkdir -p $W
echo "$(date +%T) Elmer $(command -v ElmerSolver), $SLURM_NTASKS ranks, work dir $W"
if [ ! -f $W/start0.npz ]; then
    echo "$(date +%T) 1. setup, init, transfer from $FROM"
    python3 -u -m ess setup $CFG
    python3 -u -m ess init $CFG --out $W/state0.npz
    python3 -u -m ess transfer $CFG --state $W/state0.npz --from $FROM --out $W/start0.npz
fi
if [ ! -f $W/start1.npz ]; then
    echo "$(date +%T) 2. the shelf break"
    python3 jobs/shelf_break.py $W/start0.npz $W/start1.npz --strip-divide-r-km 1520.8 --r-break-km 450 --factor 30 \
        --width-km 10 --force
fi
if [ ! -f $W/start.npz ]; then
    echo "$(date +%T) 3. the whole bed temperate"
    python3 -c "
import sys, numpy as np; sys.path.insert(0, '.')
from ess.statefile import load_state, save_state, find_name
s = load_state('$W/start1.npz')
k = find_name(s.frozen, 'ColdTempMask'); g, v = s.frozen[k]
print(f'ColdTempMask: {int((v > 0).sum())} of {len(v)} nodes temperate before; all temperate now')
s.frozen[k] = (g, np.ones_like(v))
kg = find_name(s.frozen, 'GroundedMask'); gm = s.frozen[kg][1]
print(f'GroundedMask: {int((gm > -0.5).sum())} of {len(gm)} base nodes grounded')
save_state('$W/start.npz', s.layout, s.x, s.p, s.frozen, s.diag, **s.meta)"
fi
echo "$(date +%T) 4. Newton (bed temperate), the grounding line by contact between solves"
python3 -u -m ess newton $CFG --state $W/start.npz --out $W/steady_temperate.npz
echo "$(date +%T) done"
