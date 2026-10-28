#!/bin/bash
# Periodic grounding-line test on the ElmerStreams setup: one preparation job (mesh, start state), then one job per
# treatment (one node each, 32 ranks), in parallel. Run on the login node from the repository root:
#   bash periodic_test/run_gl_ess.sh                       # elmerstreams and fix
#   MODES="elmerstreams fix none" bash periodic_test/run_gl_ess.sh
#   SCAN="10 30 100" bash periodic_test/run_gl_ess.sh      # spin-ups (fix) with fcoeff x 10, 30, 100: pick FBASE
#   FBASE=10 bash periodic_test/run_gl_ess.sh             # the full test with that base friction
#   NOPREP=1 bash periodic_test/run_gl_ess.sh             # reuse the existing mesh and start state
#   CHUNKS=1 bash periodic_test/run_gl_ess.sh             # a short trial (1 chunk = 100 yr per phase)
#   GL_CONFIG=configs/test_periodic_gl_50km.toml NTASKS=256 FBASE=10 bash periodic_test/run_gl_ess.sh
#                                                         # 50 km / 1 km across, 256 ranks (GL_CONFIG: config; NTASKS: ranks)
# Results: at the end of each run job's log (slurms/pgless_<job>.out); again with
#   sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd analyse
set -e
cd "$(dirname "$0")/.."
FROM=${FROM:-work/marine_flowline/run_cold_p268.15.npz}
[ -f "$FROM" ] || { echo "start state $FROM not found (FROM=...)"; exit 1; }
SB="sbatch --parsable ${NTASKS:+--ntasks=$NTASKS --nodes=$(( (NTASKS + 127) / 128 ))} ${NTASKS:+--time=48:00:00}"
if [ -n "$NOPREP" ]; then            # NOPREP=1: the mesh and start.npz are already there
    DEP=""
    echo "no prep job (NOPREP): using the existing mesh and start state"
else
    P=$($SB periodic_test/run_gl_ess.cmd prep --from "$FROM")
    DEP="--dependency=afterok:$P"
    echo "prep: job $P (${GL_CONFIG:-configs/test_periodic_gl.toml}; setup, init, transfer from $FROM)"
fi
if [ -n "$SCAN" ]; then
    for f in $SCAN; do
        J=$($SB $DEP periodic_test/run_gl_ess.cmd run fix --fbase $f --spinup-only ${CHUNKS:+--chunks $CHUNKS})
        echo "spin-up, fcoeff x $f: job $J (results in work/test_periodic_gl/fix_f$f)"
    done
else
    for m in ${MODES:-elmerstreams fix}; do
        J=$($SB $DEP periodic_test/run_gl_ess.cmd run $m --fbase ${FBASE:-1} ${CHUNKS:+--chunks $CHUNKS})
        echo "$m (fcoeff x ${FBASE:-1}): job $J"
    done
fi
echo "logs (each run job ends with its table): slurms/pgless_<job>.out"
