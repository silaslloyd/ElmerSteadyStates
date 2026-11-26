#!/bin/bash
# Status report of all your jobs: per job its state and what it is doing, from its own log.
#   sweep (glsweep, glsw-*)        the sweep's last step line (d, Newton iteration, rms(F), model runs so far)
#   Newton (glnt-*, glnewtherm)    its last rms(F) line
#   thermal transient (glht-*)     its last chunk line
#   FSSA land strip (log100-*)     step n/N, seconds per step and the time it finishes (jobs/eta.sh's estimate), coupled iterations of the last step
#   pending                        why (Dependency, Resources, Priority)
# Then the jobs that ended today with their state.
#   cd /SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates && bash jobs/status.sh
cd "$(dirname "$0")/.."
SETUP=${SETUP:-240}
now=$(date +%s)
echo "== $(date +%H:%M)  $(squeue -u "$USER" -h | wc -l) jobs in the queue"
squeue -u "$USER" -h -o "%i|%j|%T|%M|%D|%r" | sort -t'|' -k1,1n | while IFS='|' read J NAME ST EL ND RS; do
    printf "%s  %-30s %-8s %-9s %s node(s)" "$J" "$NAME" "$ST" "$EL" "$ND"
    [ "$ST" = PENDING ] && { echo "  [$RS]"; continue; }
    echo
    case "$NAME" in
    glsweep|glsw-*)
        L=slurms/gl_sweep_$J.out
        grep -E "sweep d=" $L 2>/dev/null | tail -1 | cut -c1-170 | sed 's/^/     /'
        ;;
    glnt-*|glnewtherm)
        grep -E "newton: " slurms/gl_newton_thermal_$J.out 2>/dev/null | tail -2 | cut -c1-170 | sed 's/^/     /' ;;
    glht-*|glheat)
        grep -E "chunk|steady enough" slurms/gl_heat_steady_$J.out 2>/dev/null | tail -2 | cut -c1-170 | sed 's/^/     /' ;;
    log100-*)
        LOG=$(ls -d work/land_flowline_3d/runs/*fssa${J}_*/elmer.log 2>/dev/null | tail -1)
        [ -n "$LOG" ] || { echo "     no run folder yet"; continue; }
        S=$(grep -E "MAIN: *Time" "$LOG" | tail -1 | sed -E 's/.*Time: *([0-9]+)\/([0-9]+):.*/\1 \2/')
        [ -n "$S" ] || { echo "     no step yet"; continue; }
        set -- $S; n=$1; N=$2
        es=$(echo "$EL" | awk -F'[-:]' '{ if (NF==2) print $1*60+$2; else if (NF==3) print $1*3600+$2*60+$3; else print $1*86400+$2*3600+$3*60+$4 }')
        dn=$((n - 1)); [ $dn -ge 1 ] || dn=1
        per=$(( (es - SETUP) / dn )); [ $per -ge 1 ] || per=1
        left=$(( (N - n + 1) * per ))
        it=$(awk '/MAIN: *Time/{n=0} /Coupled system iteration/{n=$NF} END{print n}' "$LOG")
        echo "     step $n/$N, ${per}s/step, finishes about $(date -d "@$((now + left))" +%H:%M), coupled iteration $it in this step" ;;
    esac
done
echo
echo "== ended today"
sacct -u "$USER" -S today -X -n -o JobID%9,JobName%30,State%12,ExitCode,Elapsed 2>/dev/null | grep -vE "RUNNING|PENDING" | tail -15
