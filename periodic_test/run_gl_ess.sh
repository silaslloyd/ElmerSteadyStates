#!/bin/bash
# Periodic grounding-line test on the ElmerStreams setup: one preparation job (mesh, start state), then one job per
# treatment (one node each, 32 ranks), in parallel. Run on the login node from the repository root:
#   bash periodic_test/run_gl_ess.sh                       # elmerstreams and fix
#   MODES="elmerstreams fix none" bash periodic_test/run_gl_ess.sh
#   SCAN="10 30 100" bash periodic_test/run_gl_ess.sh      # spin-ups (fix) with fcoeff x 10, 30, 100: pick FBASE
#   FBASE=30 bash periodic_test/run_gl_ess.sh             # the full test with that base friction
#   CHUNKS=1 bash periodic_test/run_gl_ess.sh             # a short trial (1 chunk = 100 yr per phase)
# Results: at the end of each run job's log (slurms/pgless_<job>.out); again with
#   sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd analyse
set -e
cd "$(dirname "$0")/.."
FROM=${FROM:-work/marine_flowline/run_cold_p268.15.npz}
[ -f "$FROM" ] || { echo "start state $FROM not found (FROM=...)"; exit 1; }
P=$(sbatch --parsable periodic_test/run_gl_ess.cmd prep --from "$FROM")
echo "prep: job $P (setup, init, transfer from $FROM)"
if [ -n "$SCAN" ]; then
    for f in $SCAN; do
        J=$(sbatch --parsable --dependency=afterok:$P periodic_test/run_gl_ess.cmd run fix --fbase $f --spinup-only ${CHUNKS:+--chunks $CHUNKS})
        echo "spin-up, fcoeff x $f: job $J (results in work/test_periodic_gl/fix_f$f)"
    done
else
    for m in ${MODES:-elmerstreams fix}; do
        J=$(sbatch --parsable --dependency=afterok:$P periodic_test/run_gl_ess.cmd run $m --fbase ${FBASE:-1} ${CHUNKS:+--chunks $CHUNKS})
        echo "$m (fcoeff x ${FBASE:-1}): job $J"
    done
fi
echo "logs (each run job ends with its table): slurms/pgless_<job>.out"
