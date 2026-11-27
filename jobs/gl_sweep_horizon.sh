#!/bin/bash
# Branches off a running sweep with a shorter Newton horizon: for each horizon in HORIZONS (yr, default "250 100": MAP_STEPS = horizon / 50
# steps of the config's dt) a sweep job (jobs/gl_sweep.cmd) from the newest converged point of the main sweep MAIN, in STEP km steps, its own
# output (sweep_gl<GL>s100_h<horizon>) and segment prefix (v, w, x, y by horizon). The steady states are the same whatever the
# horizon; what changes is the cost of one model run (it is proportional to the steps) and how much an unstable mode grows over one
# map evaluation. The first point of a branch is the main sweep's next one: compare model runs and time.
#   cd /SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates && DRY=1 bash jobs/gl_sweep_horizon.sh
#   cd /SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates && bash jobs/gl_sweep_horizon.sh
#   HORIZONS="100" bash jobs/gl_sweep_horizon.sh
# NEWTON_TOL: scaled with the horizon (5e-5 x horizon / 500 yr: 2.5e-5 at 250 yr, 1e-5 at 100 yr), because F = map(x) - x grows with the
# horizon: the same rms(F) is a 5x faster remaining drift at 100 yr than at 500 yr, so a fixed tolerance would be a looser steady state. TOL_SCALE=0: the
# fixed NEWTON_TOL (default 5e-5) for every horizon.
# GL (549), MAIN (sweep_gl549s100_25: the main sweep's folder), FROM_POINT (another start point), HORIZONS, STEP (25), DT (50), SBATCH_EXTRA, DRY=1.
set -e
cd "$(dirname "$0")/.."
GL=${GL:-549}
W=work/gl_ssa_gl$GL
MAIN=${MAIN:-sweep_gl549s100_25}
HORIZONS=${HORIZONS:-"250 100"}
STEP=${STEP:-25}
DT=${DT:-50}
FROM=${FROM_POINT:-$(ls $W/$MAIN/seg_*/points/state_*.npz | awk -F/ '{print $NF, $0}' | sort | head -1 | cut -d' ' -f2)}
[ -f "$FROM" ] || { echo "no converged point of $W/$MAIN"; exit 1; }
echo "start point $FROM (d $(python3 -c "import numpy as np; print(round(float(np.load('$FROM')['meta::sweep_d']) / 1e3, 3))") km)"
PFX=(v w x y)
i=0
for H in $HORIZONS; do
    P=${PFX[$i]}; i=$((i + 1))
    [ -n "$P" ] || { echo "at most 4 horizons"; exit 1; }
    MS=$((H / DT))
    [ $((MS * DT)) = $H ] && [ $MS -ge 1 ] || { echo "horizon $H yr is not a whole number of ${DT} yr steps"; exit 1; }
    if ls -d $W/${P}[0-9][0-9][0-9][0-9] > /dev/null 2>&1 && [ -z "$KEEP_PREFIX" ]; then
        echo "segment prefix $P is in use in $W (KEEP_PREFIX=1 to resume that sweep)"; exit 1
    fi
    if [ "${TOL_SCALE:-1}" = 0 ]; then TOL=${NEWTON_TOL:-5e-5}; else TOL=$(awk "BEGIN{print ${NEWTON_TOL:-5e-5} * $H / 500}"); fi
    CMDL="GL=$GL STEP=$STEP NEWTON_TOL=$TOL OUT=gl549s100_h$H TAGP=$P MAP_STEPS=$MS FROM_POINT=$FROM sbatch $SBATCH_EXTRA --job-name=glsw-h$H jobs/gl_sweep.cmd"
    echo "== horizon $H yr = $MS steps: Newton tolerance $TOL, prefix $P, output $W/sweep_gl549s100_h$H/"
    if [ -n "$DRY" ]; then echo "  $CMDL"; continue; fi
    J=$(GL=$GL STEP=$STEP NEWTON_TOL=$TOL OUT=gl549s100_h$H TAGP=$P MAP_STEPS=$MS FROM_POINT=$FROM sbatch --parsable $SBATCH_EXTRA --job-name=glsw-h$H jobs/gl_sweep.cmd)
    echo "  sweep job $J"
done
