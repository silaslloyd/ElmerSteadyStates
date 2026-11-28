#!/bin/bash
# On the Mac: the full ice-sheet section page (temperature, velocities: plots/plot_section.py) with a slider through the NEWTON ITERATES
# of one sweep step, rebuilt on tethys from the run cache and the sweep's log (jobs/newton_iterates.py), fetched and opened. Every
# accepted iterate is the map's output Phi(y_k) of that iterate (it differs from y_k by the residual F), on the step's own mesh.
#   cd ~/Desktop/ElmerSteadyStates && git pull && bash plots/sweep_iters_page.sh
#   D=398.9 JOB=32436 SWEEP=sweep_gl549s100_25 bash plots/sweep_iters_page.sh
# D: the step's d exactly as the log writes it ("sweep d=398.9 km": the 25 km sweep's 100 m values; the +-K sweeps print 398.93);
# JOB: the sweep's slurm job (slurms/gl_sweep_<JOB>.out); SWEEP: its folder under work/gl_ssa_gl549 (default sweep_gl549s100_25;
# +1 K sweep_gl549s100p, -1 K ...m, +2 K ...q, -2 K ...n, +3 K ...r, +4 K ...t). A second Newton of the same d (another GL round) is
# numbered on. REMOTE=0: only fetch and open.
D=${D:-398.9}
JOB=${JOB:-32436}
SWEEP=${SWEEP:-sweep_gl549s100_25}
R=/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ElmerSteadyStates
W=work/gl_ssa_gl549
SSH="ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de"
OUT=$W/iterates_${SWEEP}_$D
if [ "${REMOTE:-1}" = 1 ]; then
    $SSH slloyd@tethys-3g "bash -lc 'cd $R && git pull -q; IDX=\$(python3 -c \"print(f\\\"{int(round($D*10)):05d}\\\")\"); REF=\$(ls $W/$SWEEP/seg_*/points/state_\$IDX.npz | head -1); [ -n \"\$REF\" ] || { echo \"no point state_\$IDX.npz in $W/$SWEEP\"; exit 1; }; echo \"ref \$REF\"; rm -rf $OUT; python3 jobs/newton_iterates.py slurms/gl_sweep_$JOB.out \"sweep d=$D km\" \$REF $OUT --work $W && python3 plots/plot_section.py $OUT/iter_*.npz --out $OUT/iterates.html'" || exit 1
fi
rsync -av -e "$SSH" "slloyd@tethys-3g:$R/$OUT/iterates.html" ~/Downloads/sweep_iterates_${SWEEP}_$D.html || exit 1
open ~/Downloads/sweep_iterates_${SWEEP}_$D.html
