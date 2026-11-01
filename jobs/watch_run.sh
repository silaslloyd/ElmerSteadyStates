#!/bin/bash
# Progress of a running scaling/benchmark job, for `watch`:
#   watch -n 20 bash jobs/watch_run.sh [JOBID]        (default: your first running job)
# chunk (long runs), the step of the current Elmer run, the last Stokes / heat times per step, the GCR iterations of
# the last solves, and the job's last output line.
j=${1:-$(squeue -u $USER -h -t R -o %i | head -1)}
[ -n "$j" ] || { echo "no running job"; squeue -u $USER; exit 0; }
echo "job $j  $(squeue -h -j $j -o '%T %M (limit %l) on %D nodes' 2>/dev/null)"
f=slurms/scaling_$j.out
w=$(grep -o "runs continue from [^:]*" $f 2>/dev/null | head -1 | sed 's|runs continue from ||; s|/scaling_.*$||')
[ -n "$w" ] || w=work
L=$(ls -td $w/runs/*/ 2>/dev/null | head -1)elmer.log
grep -h "run: chunk" slurms/scaling_${j}_*.log 2>/dev/null | tail -1 | sed 's/.*run: /  /'
echo "  Elmer run $(basename $(dirname $L)): $(grep 'MAIN: Time:' $L 2>/dev/null | tail -1 | sed 's/MAIN: //')"
echo "  Stokes per step (s), last 10: $(grep 'Solver time.*stokes-vec' $L 2>/dev/null | tail -10 | awk '{printf "%.0f ", $(NF-1)}')"
# the whole step: every solver's time (Solver Timing, TIMING=1) summed between the MAIN: Time lines (complete steps only)
echo "  whole step (s, all solvers), last 10: $(awk '/MAIN: Time:/ {if (n) print t; n = 1; t = 0} /Solver time/ {t += $(NF-1)}' $L 2>/dev/null | tail -10 | awk '{printf "%.0f ", $1}')"
echo "  heat per step (s), last 5:    $(grep 'Solver time.*homologous' $L 2>/dev/null | tail -5 | awk '{printf "%.1f ", $(NF-1)}')"
echo "  GCR iterations, last 10 solves (the current one last, still counting): $(grep 'gcr:' $L 2>/dev/null | awk '$2 < prev {print prev} {prev=$2} END {print prev}' | tail -10 | tr '\n' ' ')"
m=$(grep 'NoWaterSheet.*bed nodes' $L 2>/dev/null | tail -1 | sed 's/.*bed nodes: //')
[ -n "$m" ] && echo "  mask (THERMAL=nowater): $m"
grep -q 'periodic limiter' $L 2>/dev/null && echo "  free surface: $(grep -m1 'periodic limiter' $L | sed 's/.*periodic limiter: //')"
grep -m1 -i "segmentation\|nan\b\|error termination" $L 2>/dev/null | head -1 | sed 's/^/  PROBLEM: /'
echo "  last output: $(tail -1 $f 2>/dev/null | cut -c1-100)"
