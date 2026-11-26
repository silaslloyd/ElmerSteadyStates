#!/bin/bash
# One of the implicit (adapted, coupled) FSSA land runs again with BDF order 2: the same case, step, coupled iterations, output
# frequency, 100x log-space friction pattern and start as a finished BDF 1 run (default job 32398, log100-imp-dt500: dt 500 yr x 20
# steps, 3 coupled iterations, VTU every 2 steps, 256 ranks on 4 nodes). Read from the reference run: the case and settings of its
# "FSSA test" line, the friction (the exp(a + b cos) factor of its run.sif -> middle exp(a - b), sides exp(a + b)), the sliding
# (TDepSliding in its run.sif = the config's subtemperate, otherwise temperate) and the start state (the first work/land_flowline_3d
# .npz named in its log; START=FILE to give it). Prints the sbatch line and submits it.
#
#   bash jobs/land3d_log100_bdf2.sh                     # reference job 32398
#   REF=32399 BDF=2 bash jobs/land3d_log100_bdf2.sh     # another reference run (its dt, iterations and output are copied)
#   DRY=1 bash jobs/land3d_log100_bdf2.sh               # only print
# The new run's tag is the reference's with _bdf2 (job name <reference name>-bdf2), so plots/fssa_pack.sh finds it.
set -e
cd "$(dirname "$0")/.."
REF=${REF:-32398}
BDF=${BDF:-2}
W=work/land_flowline_3d
LINE=$(grep -h "FSSA test" slurms/fssa_$REF.out | head -1)
[ -n "$LINE" ] || { echo "no 'FSSA test' line in slurms/fssa_$REF.out"; exit 1; }
TAG0=$(echo "$LINE" | sed -E 's/.*FSSA test ([^ ]+) .*/\1/')
CFG=$(echo "$LINE" | sed -E 's/.*\(([^)]+)\).*/\1/')
DT=$(echo "$LINE" | sed -E 's/.*: dt ([0-9.]+) yr x ([0-9]+) steps.*/\1/')
NST=$(echo "$LINE" | sed -E 's/.*: dt ([0-9.]+) yr x ([0-9]+) steps.*/\2/')
FSSA=$(echo "$LINE" | sed -E 's/.*FSSA ([a-z]+), coupled.*/\1/')
FSC=$(echo "$LINE" | sed -E 's/.*coupled free surface ([01]).*/\1/')
ITS=$(echo "$LINE" | sed -E 's/.*coupled iterations ([0-9]+).*/\1/')
VEV=$(echo "$LINE" | sed -E 's/.*VTU every ([0-9]+) steps.*/\1/')
NT=$(echo "$LINE" | sed -E 's/.* ([0-9]+) ranks.*/\1/')
YEARS=$(python3 -c "print(round($DT * $NST))")
NAME=$(sacct -j $REF -n -X -o JobName%40 | tr -d ' ')
D=$(ls -d $W/runs/*fssa${REF}_* | tail -1)
read YMID YSIDE < <(python3 - "$D/run.sif" <<'PY'
import math, re, sys
t = open(sys.argv[1]).read()
m = re.search(r"math\.exp\(([-0-9.eE+]+) \+ ([-0-9.eE+]+) \* math\.cos", t)
if not m:
    sys.exit("no log-space friction cosine (math.exp(a + b * math.cos)) in " + sys.argv[1])
a, b = float(m.group(1)), float(m.group(2))
print(f"{math.exp(a - b):.6g} {math.exp(a + b):.6g}")
PY
)
if grep -q "TDepSliding(" "$D/run.sif"; then SLIDING=""; else SLIDING="ESS_SLIDING=temperate"; fi
# the start: the first state file named in the reference run's log that is not its own output ($W/fssa_$TAG0.npz)
START=${START:-$(grep -h -o "$W/[A-Za-z0-9_./-]*\.npz" slurms/fssa_${REF}_*.log 2>/dev/null | grep -v "^$W/fssa_$TAG0\.npz$" | head -1)}
[ -f "$START" ] || { echo "start state of the reference run not found in its log (its own output $W/fssa_$TAG0.npz is not it): give START=FILE (the 78,000 yr state)"; exit 1; }
NNODES=$(( (NT + 63) / 64 ))
echo "reference $REF ($NAME): $CFG, FSSA $FSSA, dt $DT x $NST steps, coupled $FSC / $ITS iterations, VTU every $VEV, $NT ranks"
echo "friction: log-space cosine, middle $YMID, sides $YSIDE; sliding: ${SLIDING:-config (subtemperate)}; start $START; BDF order $BDF"
CMD="env $SLIDING ESS_SLIDE_YLOG=1 ESS_SLIDE_YMID=$YMID ESS_SLIDE_YSIDE=$YSIDE CFG=$CFG START=$START DT=$DT YEARS=$YEARS FSSA=$FSSA FS_COUPLED=$FSC COUPLED_ITERS=$ITS VTU_EVERY=$VEV BDF_ORDER=$BDF TAG=${TAG0}_bdf$BDF sbatch --nodes=$NNODES --ntasks=$NT --time=${TL:-04:00:00} --job-name=$NAME-bdf$BDF jobs/flowline_fssa.cmd"
echo "$CMD"
[ "$DRY" = 1 ] || eval "$CMD"
