#!/bin/bash
# On tethys: accuracy of the BDF 2 implicit dt 500 runs against BDF 1 (the 8-iteration, 1e-4 runs, log100): rms / max difference in Zs at
# the frame times they share (plots/fssa_compare.py), against the full-FSSA runs at dt 50 (if it exists) and dt 500, and BDF 2 against BDF 1.
# Packs the frames each run has so far first. Runs: impt_dt500 (yesterday's BDF 1, ILU3), impt_dt500_bdf1 / _bdf2 (today's, ILU2),
# impt_dt500_bdf1_ilu3 / _bdf2_ilu3 (the fixed-preconditioner pair, jobs/land3d_log100_bdf12.sh); runs that do not exist yet are left out.
#   bash plots/fssa_bdf_compare.sh
#   PACK=0 bash plots/fssa_bdf_compare.sh     # do not repack
cd "$(dirname "$0")/.."
W=work/land_flowline_3d
if [ "${PACK:-1}" = 1 ]; then bash plots/fssa_pack.sh full_dt50 impt_dt500 | grep -E "dt50"; fi
RUNS=""
for l in impt_dt500 impt_dt500_bdf1 impt_dt500_bdf2 impt_dt500_bdf1_ilu3 impt_dt500_bdf2_ilu3; do
    [ -f $W/cmp_${l}_frames.npz ] && RUNS="$RUNS $W/cmp_${l}_frames.npz"
done
[ -n "$RUNS" ] || { echo "no impt_dt500 frames files"; exit 1; }
for r in full_dt50 full_dt500; do
    [ -f $W/cmp_${r}_frames.npz ] || continue
    echo; echo "=== against $r"
    python3 plots/fssa_compare.py $W/cmp_${r}_frames.npz $RUNS 2>&1
done
echo; echo "=== against yesterday's BDF 1 (impt_dt500)"
python3 plots/fssa_compare.py $W/cmp_impt_dt500_frames.npz $(echo $RUNS | sed "s# $W/cmp_impt_dt500_frames.npz##; s#^$W/cmp_impt_dt500_frames.npz ##") 2>&1
