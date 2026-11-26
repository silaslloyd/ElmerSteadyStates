#!/bin/bash
# On the Mac: fetch the packed runs (bash plots/fssa_bdf_compare.sh on tethys first, it packs them) and open the 3D plot with the
# implicit dt 500 run with BDF 1 (imp_dt500) at the front and the BDF 2 one (imp_dt500_bdf2) beside it along y, the whole sheet, at the
# times both have (plot_state_3d.py --compare), speed coloured the same, then the same with full dt 500 against BDF 2.
#   bash plots/fssa_bdf_side.sh
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
rm -f ~/Downloads/cmp_*_frames.npz
rsync -av -e "ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de" "slloyd@tethys-3g:$D/work/land_flowline_3d/cmp_*dt500*_frames.npz" ~/Downloads/ || exit 1
cd "$(dirname "$0")/.."
A=~/Downloads/cmp_imp_dt500_frames.npz; B=~/Downloads/cmp_imp_dt500_bdf2_frames.npz; F=~/Downloads/cmp_full_dt500_frames.npz
[ -f "$A" ] && [ -f "$B" ] || { echo "$A or $B missing"; exit 1; }
echo "plot: BDF 1 | BDF 2 (implicit dt 500)"
python3 plots/plot_state_3d.py "$A" --compare "$B" --full
[ -f "$F" ] && { echo "plot: full dt 500 | BDF 2"; python3 plots/plot_state_3d.py "$F" --compare "$B" --full; }
