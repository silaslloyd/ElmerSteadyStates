#!/bin/bash
# When will each running land-strip job (log100-*) finish? From its own pace: the step it is on (n of N, from the
# "MAIN: Time: n/N" lines of its elmer.log) and its elapsed time minus SETUP seconds (default 240: the mesh and start state),
# so (elapsed - setup) / n per step, projected to N. Crude: a step is dearer early on (more coupled iterations) and the
# step it is in the middle of counts as done, so read it as an upper estimate for the implicit runs.
#   bash jobs/eta.sh          # every running log100-* job
SETUP=${SETUP:-240}
cd "$(dirname "$0")/.."
now=$(date +%s)
squeue -u "$USER" -h -t R -o "%i %j %M" | awk '$2 ~ /^log100-/' | while read J NAME EL; do
    L=$(ls -d work/land_flowline_3d/runs/*fssa${J}_*/elmer.log 2>/dev/null | tail -1)
    [ -n "$L" ] || { printf "%s %-24s no run folder yet\n" "$J" "$NAME"; continue; }
    S=$(grep -E "MAIN: *Time" "$L" | tail -1 | sed -E 's/.*Time: *([0-9]+)\/([0-9]+):.*/\1 \2/')
    [ -n "$S" ] || { printf "%s %-24s no step yet (elapsed %s)\n" "$J" "$NAME" "$EL"; continue; }
    set -- $S; n=$1; N=$2
    # elapsed "[d-][h:]m:s" to seconds
    es=$(echo "$EL" | awk -F'[-:]' '{ if (NF==2) print $1*60+$2; else if (NF==3) print $1*3600+$2*60+$3; else print $1*86400+$2*3600+$3*60+$4 }')
    done_=$((n - 1)); [ $done_ -ge 1 ] || done_=1
    per=$(( (es - SETUP) / done_ )); [ $per -ge 1 ] || per=1
    left=$(( (N - n + 1) * per ))
    printf "%s %-24s step %s/%s  %ds/step  finishes about %s\n" "$J" "$NAME" "$n" "$N" "$per" "$(date -d "@$((now + left))" +%H:%M)"
done
