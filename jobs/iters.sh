#!/bin/bash
# Coupled iterations per step of every running implicit (adapted FSSA) land-strip job, found by name (log100-imp*): the last
# few steps of each, "MAIN: Time: n/N  t: iterations" (the highest coupled iteration of that step; the cap is COUPLED_ITERS).
#   bash jobs/iters.sh          # the last 4 steps of each running job
#   bash jobs/iters.sh 10       # the last 10
#   NAMES="log100-imp" bash jobs/iters.sh      # another name prefix (default log100-imp: imp, imp8, impt, impt12)
cd "$(dirname "$0")/.."
N=${1:-4}
PRE=${NAMES:-log100-imp}
squeue -u "$USER" -h -o "%i %j" | awk -v p="$PRE" 'index($2, p) == 1' | while read J NAME; do
    L=$(ls -d work/land_flowline_3d/runs/*fssa${J}_*/elmer.log 2>/dev/null | tail -1)
    echo "== $J $NAME"
    [ -n "$L" ] || { echo "   no run folder yet"; continue; }
    awk '/MAIN: *Time/{if(n)print "   " s ": " n; s=$0; n=0} /Coupled system iteration/{n=$NF} END{print "   " s ": " n " (running)"}' "$L" | tail -$N
done
