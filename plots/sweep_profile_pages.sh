#!/bin/bash
# On the Mac: one plotly page per sweep with two plots (all the sweep points' surface profiles; all their bed temperatures), built
# on tethys over ssh (plots/sweep_profiles.py), fetched and opened. Sweeps: base (50 km first point + 25 km points), p m q n (+1 -1 +2 -2 K).
#   cd ~/Desktop/ElmerSteadyStates && bash plots/sweep_profile_pages.sh
#   REMOTE=0 bash plots/sweep_profile_pages.sh     # only fetch and open
#   ONLY="p m" bash plots/sweep_profile_pages.sh   # only these (base p m q n)
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
W=work/gl_ssa_gl549
SSH="ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de"
ONLY=${ONLY:-"base p m q n"}
if [ "${REMOTE:-1}" = 1 ]; then
    CMD="cd $D && git pull -q"
    for k in $ONLY; do
        if [ $k = base ]; then S="$W/branch_gl549s100/steady_temperate.npz $W/sweep_gl549s100/seg_s*/points/state_*.npz $W/sweep_gl549s100_25/seg_u*/points/state_*.npz"
        else S="$W/branch_gl549s100$k/steady_temperate.npz $W/sweep_gl549s100$k/seg_*/points/state_*.npz"; fi
        CMD="$CMD; python3 plots/sweep_profiles.py $S --out $W/sweep_profiles_$k.html"
    done
    $SSH slloyd@tethys-3g "bash -lc '$CMD'" || exit 1
fi
for k in $ONLY; do
    rsync -av -e "$SSH" "slloyd@tethys-3g:$D/$W/sweep_profiles_$k.html" ~/Downloads/ || exit 1
done
open $(for k in $ONLY; do echo ~/Downloads/sweep_profiles_$k.html; done)
