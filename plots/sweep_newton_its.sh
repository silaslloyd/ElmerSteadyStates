#!/bin/bash
# On the Mac: a plotly page of the Newton iterations per sweep point against d (plots/sweep_newton_its.py), built on tethys over ssh,
# fetched and opened. Default: the base sweep (268.15 K: the 50 km first segment + the 25 km sweep). Others: p m q n r t.
#   cd ~/Desktop/ElmerSteadyStates && bash plots/sweep_newton_its.sh
#   ONLY="base p m q n r t" bash plots/sweep_newton_its.sh     # several sweeps on one page
#   REMOTE=0 bash plots/sweep_newton_its.sh                    # only fetch and open
D=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
W=work/gl_ssa_gl549
SSH="ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de"
ONLY=${ONLY:-base}
if [ "${REMOTE:-1}" = 1 ]; then
    S=""
    for k in $ONLY; do
        [ -n "$S" ] && S="$S --"
        case $k in
            base) S="$S Tsurf_268.15_K(base) $W/sweep_gl549s100/seg_s*/sweep.csv $W/sweep_gl549s100_25/seg_u*/sweep.csv";;
            p) S="$S Tsurf_269.15_K $W/sweep_gl549s100p/seg_*/sweep.csv";;
            m) S="$S Tsurf_267.15_K $W/sweep_gl549s100m/seg_*/sweep.csv";;
            q) S="$S Tsurf_270.15_K $W/sweep_gl549s100q/seg_*/sweep.csv";;
            n) S="$S Tsurf_266.15_K $W/sweep_gl549s100n/seg_*/sweep.csv";;
            r) S="$S Tsurf_271.15_K $W/sweep_gl549s100r/seg_*/sweep.csv $W/sweep_gl549s100r_c/seg_*/sweep.csv";;
            t) S="$S Tsurf_272.15_K $W/sweep_gl549s100t/seg_*/sweep.csv";;
        esac
    done
    $SSH slloyd@tethys-3g "bash -lc 'cd $D && git pull -q; python3 plots/sweep_newton_its.py $S --out $W/sweep_newton_its.html'" || exit 1
fi
rsync -av -e "$SSH" "slloyd@tethys-3g:$D/$W/sweep_newton_its.html" ~/Downloads/ || exit 1
open ~/Downloads/sweep_newton_its.html
