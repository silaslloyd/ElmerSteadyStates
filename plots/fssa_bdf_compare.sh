#!/bin/bash
# On tethys: the implicit dt 500 run with BDF 2 (imp_dt500_bdf2, jobs/land3d_log100_bdf2.sh) against the BDF 1 one (imp_dt500) and
# against the full-FSSA dt 500 run: rms / max difference in Zs at the frame times they share (plots/fssa_compare.py). Packs the three
# runs first (the BDF 2 run with the frames it has so far).
#   bash plots/fssa_bdf_compare.sh
#   PACK=0 bash plots/fssa_bdf_compare.sh     # do not repack
cd "$(dirname "$0")/.."
W=work/land_flowline_3d
if [ "${PACK:-1}" = 1 ]; then bash plots/fssa_pack.sh full_dt500 imp_dt500 imp_dt500_bdf2 | grep -E "dt500"; fi
F=$W/cmp_full_dt500_frames.npz; A=$W/cmp_imp_dt500_frames.npz; B=$W/cmp_imp_dt500_bdf2_frames.npz
for f in $F $A $B; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
echo "=== BDF 1 (imp_dt500) and BDF 2 (imp_dt500_bdf2) against full dt 500"
python3 plots/fssa_compare.py $F $A $B 2>&1
echo; echo "=== BDF 2 against BDF 1 (reference: imp_dt500)"
python3 plots/fssa_compare.py $A $B 2>&1
