#!/bin/bash
# On tethys: accuracy of BDF 1 and BDF 2 (implicit dt 500, 8 iterations, tol 1e-4, log100) against the full-FSSA dt 10 run: rms / max
# difference in Zs at the frame times they share (plots/fssa_compare.py). Packs the frames each run has so far first.
# BDF 1 = impt_dt500 (yesterday's), BDF 2 = impt_dt500_bdf2 (today's, ILU2); the fixed-preconditioner pair of
# jobs/land3d_log100_bdf12.sh (impt_dt500_bdf1_ilu3, impt_dt500_bdf2_ilu3) is added when it exists.
#   bash plots/fssa_bdf_compare.sh
#   PACK=0 bash plots/fssa_bdf_compare.sh     # do not repack
cd "$(dirname "$0")/.."
W=work/land_flowline_3d
if [ "${PACK:-1}" = 1 ]; then bash plots/fssa_pack.sh full_dt10 impt_dt500 | grep -E "dt10 |impt_dt500"; fi
F=$W/cmp_full_dt10_frames.npz
[ -f $F ] || { echo "missing $F (job log100-dt10)"; exit 1; }
RUNS=""
for l in impt_dt500 impt_dt500_bdf2 impt_dt500_bdf1_ilu3 impt_dt500_bdf2_ilu3; do
    [ -f $W/cmp_${l}_frames.npz ] && RUNS="$RUNS $W/cmp_${l}_frames.npz"
done
echo; echo "=== BDF 1 and BDF 2 against full dt 10"
python3 plots/fssa_compare.py $F $RUNS 2>&1
