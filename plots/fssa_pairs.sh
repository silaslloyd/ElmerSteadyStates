#!/bin/bash
# On tethys: each implicit run against its OWN full-FSSA counterpart of the same time step (not against dt 50): rms / max
# difference in Zs at the frame times both have (plots/fssa_compare.py with the full run as the reference).
#   bash plots/fssa_pairs.sh               # pack impt, then dt 200, 500, 1000
#   bash plots/fssa_pairs.sh 500 1000      # only these time steps
#   IMP=impt12 bash plots/fssa_pairs.sh 1000   # another implicit label (imp, imp8, impt, impt12, impt20)
#   PACK=0 bash plots/fssa_pairs.sh        # do not repack first
cd "$(dirname "$0")/.."
W=work/land_flowline_3d
IMP=${IMP:-impt}
DTS="$*"; [ -n "$DTS" ] || DTS="200 500 1000"
if [ "${PACK:-1}" = 1 ]; then bash plots/fssa_pack.sh $IMP full > /dev/null; fi
for d in $DTS; do
    R=$W/cmp_full_dt${d}_frames.npz; I=$W/cmp_${IMP}_dt${d}_frames.npz
    [ -f "$R" ] && [ -f "$I" ] || { echo "dt $d: $R or $I missing"; continue; }
    echo "=== $IMP dt $d  against  full dt $d"
    python3 plots/fssa_compare.py "$R" "$I" 2>&1 | grep -v "^reference"
done
