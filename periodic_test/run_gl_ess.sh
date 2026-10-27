#!/bin/bash
# Periodic grounding-line test on the ElmerStreams setup: one preparation job (mesh, start state), then one job per
# treatment (one node each, 32 ranks), in parallel. Run on the login node from the repository root:
#   bash periodic_test/run_gl_ess.sh                       # elmerstreams and fix
#   MODES="elmerstreams fix none" bash periodic_test/run_gl_ess.sh
#   FROM=work/marine_flowline/run_cold_p268.15.npz bash periodic_test/run_gl_ess.sh
#   CHUNKS=1 bash periodic_test/run_gl_ess.sh             # a short trial (1 chunk = 100 yr per phase)
# Results: at the end of each run job's log (slurms/pgless_<job>.out); again with
#   sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd analyse
set -e
cd "$(dirname "$0")/.."
FROM=${FROM:-work/marine_flowline/run_temperate_p268.15.npz}
[ -f "$FROM" ] || { echo "start state $FROM not found (FROM=...)"; exit 1; }
P=$(sbatch --parsable periodic_test/run_gl_ess.cmd prep --from "$FROM")
echo "prep: job $P (setup, init, transfer from $FROM)"
for m in ${MODES:-elmerstreams fix}; do
    J=$(sbatch --parsable --dependency=afterok:$P periodic_test/run_gl_ess.cmd run $m ${CHUNKS:+--chunks $CHUNKS})
    echo "$m: job $J"
done
echo "logs (each run job ends with its table): slurms/pgless_<job>.out"
