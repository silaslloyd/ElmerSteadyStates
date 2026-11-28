#!/bin/bash
# On tethys: the Newton iterates of one sweep step as states (jobs/newton_iterates.py: from the run cache and the sweep's log) and the
# section page through them (plots/plot_section.py, a slider over the iterates). Called by plots/sweep_iters_page.sh.
#   bash jobs/iterates_page.sh 398.9 32436 sweep_gl549s100_25
# D: the step's d as the log writes it, JOB: the sweep's slurm job, SWEEP: its folder under work/gl_ssa_gl549.
set -e
cd "$(dirname "$0")/.."
D=$1; JOB=$2; SWEEP=$3
W=work/gl_ssa_gl549
OUT=$W/iterates_${SWEEP}_$D
IDX=$(python3 -c "print(f'{int(round($D * 10)):05d}')")
REF=$(ls $W/$SWEEP/seg_*/points/state_$IDX.npz 2>/dev/null | head -1)
[ -n "$REF" ] || { echo "no point state_$IDX.npz in $W/$SWEEP"; exit 1; }
[ -f slurms/gl_sweep_$JOB.out ] || { echo "no slurms/gl_sweep_$JOB.out"; exit 1; }
echo "ref $REF"
rm -rf $OUT
python3 jobs/newton_iterates.py slurms/gl_sweep_$JOB.out "sweep d=$D km" $REF $OUT --work $W
ls $OUT/iter_*.npz > /dev/null 2>&1 || { echo "no iterate could be rebuilt (the run cache is matched by time to the log's lines)"; exit 1; }
python3 plots/plot_section.py $OUT/iter_*.npz --out $OUT/iterates.html
