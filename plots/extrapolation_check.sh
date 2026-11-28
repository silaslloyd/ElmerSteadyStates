#!/bin/bash
# On the Mac: the extrapolation check (plots/extrapolation_check.py) on a finished sweep, built on tethys over ssh, fetched, opened.
# SWEEP: the sweep's letter (default q = +2 K, 270.15 K; p +1, r +3, t +4, m n o -1 -2 -3) or base (268.15 K).
#   cd ~/Desktop/ElmerSteadyStates && git pull && bash plots/extrapolation_check.sh
#   SWEEP=r ORDERS="1 2" NPTS=4 bash plots/extrapolation_check.sh
#   REMOTE=0 bash plots/extrapolation_check.sh     # only fetch and open
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
W=work/gl_ssa_gl549
SSH="ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de"
SWEEP=${SWEEP:-q}
ORDERS=${ORDERS:-"1 2"}
NPTS=${NPTS:-4}
OUT=$W/extrapolation_check_$SWEEP.html
if [ $SWEEP = base ]; then
    S="$W/sweep_gl549s100/seg_s*/points/state_*.npz $W/sweep_gl549s100_25/seg_u*/points/state_*.npz"
else
    S="$W/sweep_gl549s100$SWEEP/seg_*/points/state_*.npz"
fi
if [ "${REMOTE:-1}" = 1 ]; then
    $SSH slloyd@tethys-3g "bash -lc 'cd $D && git pull -q; python3 plots/extrapolation_check.py $S --orders $ORDERS --npts $NPTS --out $OUT'" || exit 1
fi
rsync -av -e "$SSH" "slloyd@tethys-3g:$D/$OUT" ~/Downloads/ || exit 1
open ~/Downloads/extrapolation_check_$SWEEP.html
