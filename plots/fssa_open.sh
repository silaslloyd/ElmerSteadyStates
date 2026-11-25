#!/bin/bash
# On the Mac: fetch the packed runs (plots/fssa_pack.sh on tethys first) and open one 3D plot per run, the whole sheet.
#   bash plots/fssa_open.sh              # all
#   bash plots/fssa_open.sh none imp     # only the runs whose label starts with one of the words (full / none / imp)
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
PRE="$*"; [ -n "$PRE" ] || PRE="full none imp"
rm -f ~/Downloads/cmp_*_frames.npz
rsync -av -e "ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de" "slloyd@tethys-3g:$D/work/land_flowline_3d/cmp_*_frames.npz" ~/Downloads/ || exit 1
cd "$(dirname "$0")/.."
for w in $PRE; do
    for f in ~/Downloads/cmp_${w}*_frames.npz; do
        [ -f "$f" ] || continue
        echo "plot $f"
        python3 plots/plot_state_3d.py "$f" --full
    done
done
