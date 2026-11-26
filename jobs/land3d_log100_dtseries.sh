#!/bin/bash
# The 8-iteration implicit land run (reference log100-impt-dt500: adapted FSSA, 8 coupled iterations, tolerance 1e-4, log100 friction, Stokes ILU3,
# output every 1000 yr) at other steps, BDF 1 and BDF 2: the BDF 1 jobs are submitted first and the BDF 2 ones with a lower priority (--nice), so BDF 1 queues first.
#   DRY=1 bash jobs/land3d_log100_dtseries.sh          # print the four sbatch lines only
#   bash jobs/land3d_log100_dtseries.sh                # submit: BDF 1 dt 100, dt 50, then BDF 2 dt 100, dt 50
#   DT_LIST="100" bash jobs/land3d_log100_dtseries.sh  # only dt 100
# 256 ranks (4 nodes) each. Time limits: dt 100 8 h, dt 50 16 h (about 2 min per dt 200 step before; smaller steps need fewer iterations).
cd "$(dirname "$0")/.."
export REFNAME=${REFNAME:-log100-impt-dt500} SS_TOL=${SS_TOL:-1e-4} STOKES_PREC=${STOKES_PREC:-ILU3}
export START=${START:-work/land_flowline_3d/state_078000.npz}
for B in 1 2; do
    for D in ${DT_LIST:-100 50}; do
        case $D in 50) TLIM=16:00:00;; 100) TLIM=08:00:00;; *) TLIM=12:00:00;; esac
        echo "=== BDF $B dt $D"
        if [ $B = 1 ]; then XTRA=""; else XTRA="--nice=100"; fi
        SBATCH_EXTRA="$XTRA" TL=$TLIM DT_NEW=$D BDF=$B bash jobs/land3d_log100_bdf2.sh
    done
done
