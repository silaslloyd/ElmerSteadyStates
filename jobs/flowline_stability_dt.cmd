#!/bin/bash
#SBATCH --output=./slurms/flowline_stability_dt_%j.out
#SBATCH --job-name=flstabdt
#SBATCH --ntasks=64
#SBATCH --time=2-00:00:00
#
# ess stability of a converged state with the physical map's step changed: [stability_map] dt = DT, n_steps = NSTEPS
# (the horizon DT x NSTEPS), on a copy of CFG; the state copied to <work>/stability_inputs/<LABEL>.npz so each run gets
# its own folder <work>/stability/<LABEL>/. For the time-step check of an oscillatory pair (backward Euler damps it
# numerically: compare DT 50 and 25 over the same horizon).
#   CFG=... STATE=... DT=25 NSTEPS=100 LABEL=u0700_dt25 sbatch jobs/flowline_stability_dt.cmd
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
W=work/flowline_rect_coarse
: "${CFG:?CFG}" "${STATE:?STATE}" "${DT:?DT}" "${NSTEPS:?NSTEPS}" "${LABEL:?LABEL}"
[ -f "$STATE" ] || { echo "no state $STATE"; exit 1; }
C=${CFG%.toml}_${LABEL}.toml
awk -v dt=$DT -v n=$NSTEPS '
    /^\[/ { sec = $0 }
    sec == "[stability_map]" && /^dt *=/ { print "dt = " dt ".0"; next }
    sec == "[stability_map]" && /^n_steps *=/ { print "n_steps = " n; next }
    { print }' $CFG > $C
mkdir -p $W/stability_inputs
S=$W/stability_inputs/$LABEL.npz
[ -f $S ] || cp $STATE $S
echo "$(date +%T) stability of $STATE ($S) on $C: [stability_map] $NSTEPS x $DT yr, m = ${M:-30}"
grep -A4 "^\[stability_map\]" $C
python3 -u -m ess stability $C --state $S --m ${M:-30}
echo "$(date +%T) done"
