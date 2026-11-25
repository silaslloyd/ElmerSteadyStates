#!/bin/bash
# On the Mac: fetch the packed runs (plots/fssa_pack.sh impt on tethys first) and open one 3D plot per time step with the
# implicit run (impt) beside its full-FSSA counterpart, the whole sheet, at the times both have (plot_state_3d.py --compare:
# the first file at the front, the second beside it along y, speed coloured the same).
#   bash plots/fssa_side.sh              # dt 200, 500 and 1000
#   bash plots/fssa_side.sh 500 1000     # only these
#   IMP=imp bash plots/fssa_side.sh 500  # the 3-iteration implicit run (imp), or imp8, as the first label
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
DTS="$*"; [ -n "$DTS" ] || DTS="200 500 1000"
IMP=${IMP:-impt}
rm -f ~/Downloads/cmp_*_frames.npz
rsync -av -e "ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de" "slloyd@tethys-3g:$D/work/land_flowline_3d/cmp_*_frames.npz" ~/Downloads/ || exit 1
cd "$(dirname "$0")/.."
for d in $DTS; do
    F=~/Downloads/cmp_full_dt${d}_frames.npz
    I=~/Downloads/cmp_${IMP}_dt${d}_frames.npz
    [ -f "$F" ] && [ -f "$I" ] || { echo "dt $d: $F or $I missing"; continue; }
    echo "plot dt $d: full | $IMP"
    python3 plots/plot_state_3d.py "$F" --compare "$I" --full
done
