#!/bin/bash
# On the Mac, all of it: packs the runs' frames and builds the CPU time page on tethys over ssh, fetches both, and opens the two
# plotly pages:
#   1. fssa_rel_rms.html   relative rms of every run against full dt 10, at each output time (full dashed, implicit solid, BDF 2 squares)
#   2. fssa_cpu_time.html  CPU time (core-hours) of the finished 10,000 yr runs against the time step: full, implicit BDF 1, implicit BDF 2
#   cd ~/Desktop/ElmerSteadyStates && bash plots/fssa_results.sh
#   REMOTE=0 bash plots/fssa_results.sh      # skip the tethys step (already packed there)
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
SSH="ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de"
if [ "${REMOTE:-1}" = 1 ]; then
    $SSH slloyd@tethys-3g "bash -lc 'cd $D && git pull -q; bash plots/fssa_pack.sh > /dev/null && python3 plots/fssa_cpu_time.py --out work/land_flowline_3d/fssa_cpu_time.html'" || exit 1
fi
rm -f ~/Downloads/cmp_*_frames.npz
rsync -av -e "$SSH" "slloyd@tethys-3g:$D/work/land_flowline_3d/cmp_*_frames.npz" ~/Downloads/ || exit 1
rsync -av -e "$SSH" "slloyd@tethys-3g:$D/work/land_flowline_3d/fssa_cpu_time.html" ~/Downloads/ || exit 1
cd "$(dirname "$0")/.."
python3 plots/fssa_rel_rms.py ~/Downloads --out ~/Downloads/fssa_rel_rms.html && open ~/Downloads/fssa_rel_rms.html ~/Downloads/fssa_cpu_time.html
