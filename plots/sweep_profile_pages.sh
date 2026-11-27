#!/bin/bash
# On the Mac: ONE plotly page with two plots (all the sweep points' surface profiles; all their bed temperatures at the ice base) and a
# slider over the sweep's parameter value, the surface temperature Tsurf. Built on tethys over ssh (plots/sweep_profiles.py), fetched, opened.
# Sweeps: base (50 km first point + 25 km points), p m q n r t (+1 -1 +2 -2 +3 +4 K); one with no states yet is left out.
#   cd ~/Desktop/ElmerSteadyStates && bash plots/sweep_profile_pages.sh
#   REMOTE=0 bash plots/sweep_profile_pages.sh     # only fetch and open
#   ONLY="base p m" bash plots/sweep_profile_pages.sh   # only these sweeps
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
W=work/gl_ssa_gl549
SSH="ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de"
ONLY=${ONLY:-"base p m q n r t"}
if [ "${REMOTE:-1}" = 1 ]; then
    S=""
    for k in $ONLY; do
        [ -n "$S" ] && S="$S --"
        if [ $k = base ]; then S="$S $W/branch_gl549s100/steady_temperate.npz $W/sweep_gl549s100/seg_s*/points/state_*.npz $W/sweep_gl549s100_25/seg_u*/points/state_*.npz"
        else S="$S $W/branch_gl549s100$k/steady_temperate.npz $W/sweep_gl549s100$k/seg_*/points/state_*.npz"; fi
    done
    $SSH slloyd@tethys-3g "bash -lc 'cd $D && git pull -q; python3 plots/sweep_profiles.py $S --out $W/sweep_profiles.html'" || exit 1
fi
rsync -av -e "$SSH" "slloyd@tethys-3g:$D/$W/sweep_profiles.html" ~/Downloads/ || exit 1
open ~/Downloads/sweep_profiles.html
