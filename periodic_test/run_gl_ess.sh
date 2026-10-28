#!/bin/bash
# Periodic grounding-line test on the ElmerStreams setup: one preparation job (mesh, start state), then one job per
# treatment and friction heating (one node each, 32 ranks), in parallel. Run on the login node from the repository root:
#   bash periodic_test/run_gl_ess.sh                       # elmerstreams_loads, fix_loads, fix_flux
#   RUNS="fix_loads none_loads" bash periodic_test/run_gl_ess.sh   # others (MODE_HEAT)
#   SCAN="10 30 100" bash periodic_test/run_gl_ess.sh      # spin-ups (fix) with fcoeff x 10, 30, 100: pick FBASE
#   FBASE=30 bash periodic_test/run_gl_ess.sh             # another base friction (default 10)
#   NOPREP=1 bash periodic_test/run_gl_ess.sh             # reuse the existing mesh and start state
#   CHUNKS=1 bash periodic_test/run_gl_ess.sh             # a short trial (1 chunk = 100 yr per phase)
#   RESUME=spinup SPINUP_MAX=200 NOPREP=1 bash periodic_test/run_gl_ess.sh   # continue the runs from their last saved
#                                                         # spin-up chunk (then retreat, advance); SPINUP_MAX: chunk cap
#   GL_CONFIG=configs/test_periodic_gl_50km.toml NTASKS=256 FBASE=10 bash periodic_test/run_gl_ess.sh
#                                                         # 50 km / 1 km across, 256 ranks (GL_CONFIG: config; NTASKS: ranks)
# Results: at the end of each run job's log (slurms/pgless_<job>.out); again with
#   sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd analyse
set -e
cd "$(dirname "$0")/.."
FROM=${FROM:-work/marine_flowline/run_cold_p268.15.npz}
[ -f "$FROM" ] || { echo "start state $FROM not found (FROM=...)"; exit 1; }
SB="sbatch --parsable ${NTASKS:+--ntasks=$NTASKS --nodes=$(( (NTASKS + 127) / 128 ))} ${NTASKS:+--time=48:00:00}"
# the Elmer libraries first, in one job (parallel jobs building at once break each other's builds)
B=$(sbatch --parsable --ntasks=1 --time=00:20:00 periodic_test/run_gl_ess.cmd build)
DEP="--dependency=afterok:$B"
echo "build: job $B"
if [ -n "$NOPREP" ]; then            # NOPREP=1: the mesh and start.npz are already there
    echo "no prep job (NOPREP): using the existing mesh and start state"
else
    P=$($SB $DEP periodic_test/run_gl_ess.cmd prep --from "$FROM")
    DEP="--dependency=afterok:$P"
    echo "prep: job $P (${GL_CONFIG:-configs/test_periodic_gl.toml}; setup, init, transfer from $FROM)"
fi
if [ -n "$SCAN" ]; then
    for f in $SCAN; do
        J=$($SB $DEP periodic_test/run_gl_ess.cmd run fix --fbase $f --spinup-only ${CHUNKS:+--chunks $CHUNKS})
        echo "spin-up, fcoeff x $f: job $J (results in work/test_periodic_gl/fix_f$f)"
    done
else
    # RUNS: MODE_HEAT; default the three of the loads test: ElmerStreams as it is (halved raw loads, contact with
    # Fw2), the fix with the loads heating, the fix with the flux heating (the reference)
    for r in ${RUNS:-elmerstreams_loads fix_loads fix_flux}; do
        m=${r%_*}; h=${r##*_}
        J=$($SB $DEP periodic_test/run_gl_ess.cmd run $m --heat $h --fbase ${FBASE:-10} ${CHUNKS:+--chunks $CHUNKS} \
              ${RESUME:+--resume $RESUME} ${SPINUP_MAX:+--spinup-max $SPINUP_MAX})
        echo "$m, friction heating $h (fcoeff x ${FBASE:-10}): job $J (results in ${m}_$h)"
    done
    echo "then: sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd compare fix_loads fix_flux"
    echo "      sbatch --ntasks=1 --time=00:30:00 periodic_test/run_gl_ess.cmd compare elmerstreams_loads fix_flux"
fi
echo "logs (each run job ends with its table): slurms/pgless_<job>.out"
