#!/bin/bash
#SBATCH --output=./slurms/gl_stability_%j.out
#SBATCH --job-name=glstab
#SBATCH --ntasks=128
#SBATCH --time=3-00:00:00
#
# Linear stability (ess stability: Arnoldi on the PHYSICAL map [stability_map] linearised at the state, M model runs) of a converged
# marine sweep point, on the segment's own mesh and config. Both masks are frozen in the map (the temperate patch and the grounding line,
# as in the sweep's Newton): the stability of the state with the patch and the grounding line held. Logs the leading |mu| as Arnoldi
# goes on; |mu| > 1 (sigma = ln(mu)/(map time) > 0) is unstable. Output <work>/stability/<state name>/ (stability.npz; arnoldi.npz is a
# checkpoint: resubmitting the same STATE resumes). One product is the whole [stability_map] (e.g. 100 steps x 50 yr): long.
#   CFG=work/gl_ssa_gl549/refined/j0000.toml STATE=work/gl_ssa_gl549/sweep_gl549s100r/seg_j0000/points/state_05239.npz M=20 sbatch jobs/gl_stability.cmd
# CFG, STATE (required), M (Krylov dimension, default [stability] m), MAP (stability_map | map).
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
[ -n "$CFG" ] && [ -f "$CFG" ] || { echo "CFG not set or no such file: $CFG"; exit 1; }
[ -n "$STATE" ] && [ -f "$STATE" ] || { echo "STATE not set or no such file: $STATE"; exit 1; }
echo "$(date +%T) stability of $STATE on $CFG, map ${MAP:-stability_map}, m = ${M:-config}"
echo "  [stability_map] of the config:"; grep -A6 "^\[stability_map\]" "$CFG" | sed 's/^/    /' || echo "    (none: the [map] is used)"
python3 -u -m ess stability "$CFG" --state "$STATE" --map ${MAP:-stability_map} $( [ -n "$M" ] && echo "--m $M" )
echo "$(date +%T) done"
