#!/bin/bash
# Follow one of the bounds jobs (jobs/flowline_bounds.cmd):  bash jobs/watch_bounds.sh cold | temperate
#   the job's chunk lines (rms change, grounding line) from its slurm log, then the current Elmer run's step and its
#   two Stokes iterations, refreshed every 30 s (Ctrl-C to stop)
cd "$(dirname "$0")/.."
which=${1:-cold}
case $which in
    cold) pat='fcoeff * (ess_slide_maxfactor' ;;
    temperate) pat='fcoeff * ess_grounded(tx[2])' ;;
    *) echo "usage: bash jobs/watch_bounds.sh cold | temperate"; exit 1 ;;
esac
while true; do
    S=$(grep -l "$which bound:" $(ls -t slurms/flowline_bounds_*.out) 2>/dev/null | head -1)
    D=""
    for d in $(ls -td work/flowline_rect_ext/runs/*/ 2>/dev/null | head -40); do
        if [ -f $d/run.sif ] && grep -qF "$pat" $d/run.sif; then D=$d; break; fi
    done
    clear
    echo "=== $which bound   $(date +%T)   slurm: $S"
    [ -n "$S" ] && grep "bound:\|run: chunk\|grounding line\|Error\|Traceback\|ModelFailure" $S | tail -12
    if [ -n "$D" ]; then
        L=$D/elmer.log
        echo "=== Elmer: $L"
        grep "Time:" $L | tail -1
        grep "stokes-vec" $L | tail -2
        grep -i "error" $L | tail -3
    fi
    sleep 30
done
