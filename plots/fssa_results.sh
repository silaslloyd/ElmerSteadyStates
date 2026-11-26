#!/bin/bash
# On the Mac: the two FSSA plotly pages, opened. Run on tethys first (packs the runs' frames, builds the CPU time page):
#   cd /SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates && bash plots/fssa_pack.sh && python3 plots/fssa_cpu_time.py --out work/land_flowline_3d/fssa_cpu_time.html
# then here:
#   cd ~/Desktop/ElmerSteadyStates && bash plots/fssa_results.sh
#   1. fssa_rel_rms.html   relative rms of every run against full dt 10, at each output time (full dashed, implicit solid, BDF 2 squares)
#   2. fssa_cpu_time.html  CPU time (core-hours) of the finished 10,000 yr runs against the time step: full, implicit BDF 1, implicit BDF 2
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
rm -f ~/Downloads/cmp_*_frames.npz
rsync -av -e "ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de" "slloyd@tethys-3g:$D/work/land_flowline_3d/cmp_*_frames.npz" "slloyd@tethys-3g:$D/work/land_flowline_3d/fssa_cpu_time.html" ~/Downloads/ || exit 1
cd "$(dirname "$0")/.."
python3 plots/fssa_rel_rms.py ~/Downloads --out ~/Downloads/fssa_rel_rms.html && open ~/Downloads/fssa_rel_rms.html ~/Downloads/fssa_cpu_time.html
