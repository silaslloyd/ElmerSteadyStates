#!/bin/bash
# On the Mac: fetch every packed run (bash plots/fssa_pack.sh on tethys first) and open the relative rms page: every full-FSSA run (dashed)
# and every implicit run (solid, BDF 2 with squares) against the full dt 10 run, at each output time.
#   bash plots/fssa_rel_rms.sh
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
rm -f ~/Downloads/cmp_*_frames.npz
rsync -av -e "ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de" "slloyd@tethys-3g:$D/work/land_flowline_3d/cmp_*_frames.npz" ~/Downloads/ || exit 1
cd "$(dirname "$0")/.."
python3 plots/fssa_rel_rms.py ~/Downloads --out ~/Downloads/fssa_rel_rms.html && open ~/Downloads/fssa_rel_rms.html
