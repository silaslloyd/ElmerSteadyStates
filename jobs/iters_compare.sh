#!/bin/bash
# Coupled iterations per step of two land-strip runs side by side, by job id (running or finished).
#   bash jobs/iters_compare.sh 32407 32431      # left: first job, right: second job
cd "$(dirname "$0")/.."
A=$1; B=$2
get() {
    L=$(ls -d work/land_flowline_3d/runs/*fssa${1}_*/elmer.log 2>/dev/null | tail -1)
    [ -n "$L" ] || return
    awk '/MAIN: *Time/{if(s!="")print s, n; s=$0; sub(/.*Time: */, "", s); sub(/\/.*/, "", s); n=0}
         /Coupled system iteration/{n=$NF}
         END{if(s!="")print s, n}' "$L"
}
join -a1 -a2 -e "-" -o 0,1.2,2.2 <(get "$A" | sort -k1,1) <(get "$B" | sort -k1,1) | sort -n \
  | awk -v a="$A" -v b="$B" 'BEGIN{printf "step  %8s  %8s\n", a, b} {printf "%4s  %8s  %8s\n", $1, $2, $3}'
