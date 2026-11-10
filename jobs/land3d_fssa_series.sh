#!/bin/bash
# Implicit (adapted) FSSA against full FSSA on a 3D land ice sheet: no grounding line, no shelf, no SSA, no heat
# (temperature frozen; Stokes and the top free surface only). configs/land_margin_3d.toml: the land-terminating
# margin on a 20 km strip with free-slip side walls, 1 node (jobs/flowline_fssa.cmd).
#
# The start is the case's start state with a Gaussian surface bump (periodic_test/perturb_surface.py: 100 m, radius
# 4 km, mid-strip half way to the margin), so the flow is 3D and the bump's spreading is the stiff free-surface
# motion that limits the step. Every run covers YEARS (default 1000) with boundary VTUs every 100 yr.
#
#   bash jobs/land3d_fssa_series.sh            (from the repo root, on the login node)
#
# Two lanes, each a chain (at most two jobs at once):
#   lane A: full dt5 (the reference) -> full dt20 -> 50 -> 100 -> 200 -> 500
#   lane B: adapted dt20 -> 50 -> 100 -> 200 -> 500          (3 coupled iterations)
# Settings: YEARS, DTS (default "20 50 100 200 500"), AMP / RADIUS (the bump), TL (time limit, default 12 h),
# AFTER_A / AFTER_B (each lane waits for that job first), SUFFIX (added to the tags).
# Compare:  python3 plots/mech_compare.py <ref id> <ids ...> --plot land3d.png
set -e
cd "$(dirname "$0")/.."
CFG=configs/land_margin_3d.toml
W=work/land_margin_3d
YEARS=${YEARS:-1000}; DTS=${DTS:-20 50 100 200 500}; TL=${TL:-12:00:00}
START=$W/fssa_start_bump${SUFFIX:-}.npz
if [ ! -f $W/fssa_start.npz ]; then
    echo "$W/fssa_start.npz missing: run one setup job first, e.g."
    echo "  CFG=$CFG DT=50 YEARS=100 sbatch jobs/flowline_fssa.cmd"
    exit 1
fi
[ -f $START ] || python3 periodic_test/perturb_surface.py $W/fssa_start.npz $START --amp ${AMP:-100} --radius ${RADIUS:-4}

# one run (DT FSSA EXTRA AFTER) -> job id
run1() { env CFG=$CFG START=$START DT=$1 YEARS=$YEARS FSSA=$2 ${3:+$3} VTU_EVERY=$(( 100 / $1 > 0 ? 100 / $1 : 1 )) \
       TAG=land3d_bump_dt$1_$2${SUFFIX:-} sbatch --parsable --time=$TL ${4:+--dependency=afterany:$4} jobs/flowline_fssa.cmd; }

echo "3D land FSSA series: ${YEARS} yr per run, VTU every 100 yr, start $START"
a=$(run1 5 full "" "${AFTER_A:-}"); echo "  A full dt5 (reference): $a"
for dt in $DTS; do a=$(run1 $dt full "" $a); echo "  A full dt$dt: $a"; done
b=${AFTER_B:-}
for dt in $DTS; do b=$(run1 $dt adapted "FS_COUPLED=1 COUPLED_ITERS=3" $b); echo "  B adapted dt$dt: $b"; done
echo "submitted; squeue -u \$USER -o \"%.8i %.10T %E\" shows the chains"
