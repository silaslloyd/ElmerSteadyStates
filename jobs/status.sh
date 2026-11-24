#!/bin/bash
# Progress of all your running / pending jobs in one go:  bash jobs/status.sh
# For each job: name, state, elapsed, then the last lines of its slurm output that show progress
# (chunk / grounding line / newton / BRANCH / step lines); for the 3D fssa runs also the newest Elmer log step and Stokes iteration.
# N=<lines per job> (default 3)
cd "$(dirname "$0")/.." || exit 1
N=${N:-3}
echo "=== $(date +%H:%M:%S)  jobs of $USER ==="
squeue -u "$USER" -h -o "%i|%j|%T|%M|%N|%r" | while IFS='|' read -r id name state el nodes why; do
    echo
    echo "--- $id  $name  $state  $el  ${nodes:+nodes $nodes}${why:+($why)}"
    [ "$state" = PENDING ] && continue
    f=$(ls -t slurms/*_"$id".out 2>/dev/null | head -1)
    [ -z "$f" ] && f=$(ls -t slurms/*"$id"*.out 2>/dev/null | head -1)
    if [ -z "$f" ]; then echo "  (no slurm output file found)"; continue; fi
    echo "  file: $f  (last written $(date -r "$f" +%H:%M:%S))"
    grep -E "BRANCH [^ ]+: .*(batch|drift|rms|grounding)|S[0-9][a-z]?[: ].*(rms|done|stop|converged)|chunk [0-9]+/|contact grounding line|newton: it|rms\(F\)|stage|FSSA|step [0-9]+" "$f" \
        | cut -c1-230 | tail -n "$N" | sed 's/^/  /'
    [ "$N" -gt 0 ] && [ -z "$(grep -E 'chunk|newton|BRANCH|step' "$f" | tail -1)" ] && tail -n 2 "$f" | cut -c1-230 | sed 's/^/  /'
    case "$name" in
    f3d-*)  # the 3D free-surface runs: the Elmer log of the newest run directory
        # the run directory carries the job label (fssa<jobid>); else the newest log of the 3D work dirs only
        lg=$(ls -t work/*/runs/*fssa"$id"*/elmer.log 2>/dev/null | head -1)
        [ -z "$lg" ] && lg=$(ls -t work/flowline_gl_3d*/runs/*/elmer.log 2>/dev/null | head -1)
        if [ -n "$lg" ]; then
            echo "  elmer log: $lg"
            grep "MAIN: Time" "$lg" | tail -1 | sed 's/^/  /'
            grep -E "gcr: " "$lg" | tail -1 | sed 's/^/  last Stokes iteration: /'
            grep "Coupled system iteration" "$lg" | tail -1 | sed 's/^/  /'
        fi;;
    esac
done
echo
echo "=== recent grounding line moves in the 125 m branch ==="
grep -h "BRANCH r125" slurms/gl_branch_*.out 2>/dev/null | tail -3 | cut -c1-230
