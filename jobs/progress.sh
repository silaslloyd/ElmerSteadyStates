#!/bin/bash
# Progress of all your running jobs in one go (and what is queued):
#   bash jobs/progress.sh
# Land FSSA runs: the step reached of the total, and the surface change of the last step; grounding-line branch and
# Newton jobs: the last log line of their slurm file; others: the name and elapsed time only.
cd "$(dirname "$0")/.."
echo "== running"
squeue -u "$USER" -h -t R -o "%i|%j|%M|%N" | while IFS='|' read J N T NODES; do
    echo "-- $J $N  ($T on $NODES)"
    L=$(ls -d work/land_flowline_3d/runs/*fssa${J}_* 2>/dev/null | tail -1)
    if [ -n "$L" ] && [ -f "$L/elmer.log" ]; then
        grep -E "MAIN: *Time" "$L/elmer.log" | tail -1 | sed 's/^/   step: /'
        grep "ComputeChange" "$L/elmer.log" | tail -1 | cut -c1-150 | sed 's/^/   change: /'
    else
        F=$(ls slurms/*_${J}.out 2>/dev/null | tail -1)
        [ -n "$F" ] && tail -2 "$F" | cut -c1-200 | sed 's/^/   /'
    fi
done
echo "== queued"
squeue -u "$USER" -t PD -o "%i %j %R %S"
