#!/bin/bash
#SBATCH --output=./slurms/flowline_check_dt_%j.out
#SBATCH --job-name=flchkdt
#SBATCH --ntasks=64
#SBATCH --time=0-03:00:00
#
# Does Newton's map need 40 x 50 yr? Its fixed points are the steady states whatever the step (implicit steps: a state
# that does not change over a step satisfies the steady equations), so fewer, longer steps over the same 2000 yr
# horizon would make every map run (every GMRES product) cheaper. This checks it on a converged sweep point: the
# residual rms(Phi(x) - x) with the map as now and with DTS (default "100 200": 20 x 100 yr, 10 x 200 yr). If the
# residuals stay at the converged level (~5e-6) and the runs succeed, the longer steps give the same steady states;
# the log's timestamps give each run's cost.
#   STATE=... CFG=... DTS="100 200" sbatch jobs/flowline_check_dt.cmd
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
CFG=${CFG:-$W/refined/m0545.toml}
STATE=${STATE:-$W/sweep_refined/seg_m0545/points/state_04942.npz}
HORIZON=2000
for DT in 50 ${DTS:-100 200}; do
    N=$(awk -v h=$HORIZON -v d=$DT 'BEGIN { printf "%d", h / d }')
    C=${CFG%.toml}_dt$DT.toml
    awk -v dt=$DT -v n=$N '
        /^\[/ { sec = $0 }
        sec == "[map]" && /^dt *=/ { print "dt = " dt ".0"; next }
        sec == "[map]" && /^n_steps *=/ { print "n_steps = " n; next }
        { print }' $CFG > $C
    echo "$(date +%T) ===== map $N x $DT yr ($C) on $STATE"
    python3 -u -m ess check $C --state $STATE --skip-roundtrip --eps 1e-6 --vector residual 2>&1 \
        | grep -E "check |block |linear|eps|Error|error|failed|FAILED" || true
    echo "$(date +%T) ===== done $N x $DT yr"
done
