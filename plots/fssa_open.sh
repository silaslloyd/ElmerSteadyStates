#!/bin/bash
# On the Mac: fetch the packed runs (plots/fssa_pack.sh on tethys first) and open one 3D plot per run, the whole sheet.
#   bash plots/fssa_open.sh
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
rm -f ~/Downloads/cmp_*_frames.npz
rsync -av -e "ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de" "slloyd@tethys-3g:$D/work/land_flowline_3d/cmp_*_frames.npz" ~/Downloads/ || exit 1
cd "$(dirname "$0")/.."
for f in ~/Downloads/cmp_*_frames.npz; do
    echo "plot $f"
    python3 plots/plot_state_3d.py "$f" --full
done
