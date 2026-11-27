#!/bin/bash
# On the Mac, all of it: builds the sweep section pages (plots/plot_section.py: ice temperature section, switchable velocity fields,
# draggable head pointer) on tethys over ssh, fetches them and opens them. One page per sweep:
#   base (268.15 K)  sweep_gl549s100 (50 km first point) + sweep_gl549s100_25
#   +1 K  sweep_gl549s100p   -1 K  sweep_gl549s100m   +2 K  sweep_gl549s100q   -2 K  sweep_gl549s100n
#   cd ~/Desktop/ElmerSteadyStates && bash plots/sweep_pages.sh
#   REMOTE=0 bash plots/sweep_pages.sh       # only fetch and open (pages built already)
#   ONLY="p m" bash plots/sweep_pages.sh     # only these (base p m q n)
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
W=work/gl_ssa_gl549
SSH="ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de"
ONLY=${ONLY:-"base p m q n"}
if [ "${REMOTE:-1}" = 1 ]; then
    CMD="cd $D && git pull -q"
    for k in $ONLY; do
        if [ $k = base ]; then S="$W/branch_gl549s100/steady_temperate.npz $W/sweep_gl549s100/seg_s*/points/state_*.npz $W/sweep_gl549s100_25/seg_u*/points/state_*.npz"
        else S="$W/branch_gl549s100$k/steady_temperate.npz $W/sweep_gl549s100$k/seg_*/points/state_*.npz"; fi
        CMD="$CMD; python3 plots/plot_section.py $S --out $W/sweep_sections_$k.html"
    done
    $SSH slloyd@tethys-3g "bash -lc '$CMD'" || exit 1
fi
for k in $ONLY; do
    rsync -av -e "$SSH" "slloyd@tethys-3g:$D/$W/sweep_sections_$k.html" ~/Downloads/ || exit 1
done
open $(for k in $ONLY; do echo ~/Downloads/sweep_sections_$k.html; done)
