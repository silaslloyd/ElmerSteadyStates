#!/bin/bash
#SBATCH --output=./slurms/gl_bisect_%j.out
#SBATCH --job-name=glbis
#SBATCH --ntasks=128
#SBATCH --time=1-00:00:00
#
# The bisection of one interval of a sweep where the consistency of the steady state changes (no cold bed above melting and the patch's
# water sheet above 0), as a job of its own: started by `ess sweep --bisect-spawn` (jobs/gl_sweep.cmd) with the interval in the environment,
# so the sweep goes on while this runs. Midpoints solved from the consistent-side state down to BIS_KM; they go to BIS_OUT/bisect.csv and
# BIS_OUT/points/ (the sweep's own sweep.csv is not touched).
#   BIS_CFG   the segment's config (refined/<tag>.toml, its mesh and map steps)
#   BIS_A, BIS_AD, BIS_ACONS   the state at one end (file, d in km, 1 consistent / 0 not)
#   BIS_B, BIS_BD, BIS_BCONS   the other end
#   BIS_OUT   output folder, BIS_KM   where to stop (km)
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
export ESS_NEWTON_TOL=${ESS_NEWTON_TOL:-${NEWTON_TOL:-5e-5}}      # the sweep's (it exports ESS_NEWTON_TOL)
for v in BIS_CFG BIS_A BIS_AD BIS_ACONS BIS_B BIS_BD BIS_BCONS BIS_OUT BIS_KM; do
    [ -n "${!v}" ] || { echo "$v not set: this job is started by ess sweep --bisect-spawn"; exit 1; }
done
echo "$(date +%T) bisection d = $BIS_AD (consistent $BIS_ACONS) .. $BIS_BD km (consistent $BIS_BCONS) to $BIS_KM km; config $BIS_CFG; out $BIS_OUT"
python3 -u -m ess sweep "$BIS_CFG" --state "$BIS_A" --d-start-km "$BIS_AD" --d-end-km "$BIS_BD" --bisect-km "$BIS_KM" \
    --bisect-only "$BIS_B,$BIS_BD,$BIS_ACONS,$BIS_BCONS" --out "$BIS_OUT"
echo "$(date +%T) done"
