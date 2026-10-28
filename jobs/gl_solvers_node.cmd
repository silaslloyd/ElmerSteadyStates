#!/bin/bash
#SBATCH --output=./slurms/glsolvers_%j.out
#SBATCH --job-name=glsolvers
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#
# The moving-grounding-line solver test (configs/test_gl_solvers.toml) with ALL solvers at the same time on ONE
# node: each solver run gets RANKS MPI ranks (default 32; 4 runs x 32 = the node's 128 cores), unpinned so the
# runs share the node. Submitted by jobs/gl_solvers.sh (which sets --ntasks); by hand:
#   sbatch --ntasks=128 --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,SOLVERS="direct iterative block bubbles" \
#       jobs/gl_solvers_node.cmd
# GL_CONFIG: the config (default configs/test_gl_solvers.toml). Builds the libraries, makes the mesh and start
# state if <work dir>/start.npz is missing (partitioned
# for RANKS), runs the solvers in parallel (logs slurms/glsolvers_<job>_<solver>.log), then prints the comparison.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
SOLVERS=${SOLVERS:-direct iterative block}
RANKS=${RANKS:-32}
export GL_CONFIG=${GL_CONFIG:-configs/test_gl_solvers.toml}
export ELMER_MODULE=${ELMER_MODULE:?ELMER_MODULE not set}
FROM=${FROM:-work/marine_flowline/run_cold_p268.15.npz}
L=slurms/glsolvers_${SLURM_JOB_ID:-local}

bash periodic_test/run_gl_ess.cmd build                     # the libraries for this Elmer (module load inside)
module load "$ELMER_MODULE"
echo "Elmer: $ELMER_MODULE -> $(command -v ElmerSolver); $RANKS ranks per solver: $SOLVERS"
export ESS_NPROCS=$RANKS                                    # every Elmer run (and the mesh partitioning) uses RANKS
export ESS_MPIRUN="mpirun -np {n} --bind-to none"           # the runs share the node's cores
WORK=$(python3 -c "import sys; sys.path.insert(0, 'periodic_test'); import run_gl_ess as r; print(r.WORK)" 2>/dev/null | tail -1)
echo "config $GL_CONFIG, work dir $WORK"
if [ ! -f "$WORK/start.npz" ]; then
    echo "$(date +%T) prep: mesh and start state (from $FROM), log ${L}_prep.log"
    python3 -u periodic_test/run_gl_ess.py prep --from "$FROM" > ${L}_prep.log 2>&1
fi
# START=<state>: start every run from it (not <work dir>/start.npz); TAG_PREFIX: results in <prefix><solver>;
# SPINUP_TOL / SPINUP_MAX / SPINUP_ONLY=1: the spin-up's tolerance, chunk cap, and spin-up only
# RESUME=retreat: continue each run from its finished spin-up (its last saved state) with the later phases
# PHASES="retreat=50,1,0 advance=50,1,3": phase overrides NAME=CHUNKS,FRICTION,ACCUMULATION (run_gl_ess.py --phase)
PHASE_ARGS=""
for ph in ${PHASES:-}; do PHASE_ARGS="$PHASE_ARGS --phase $ph"; done
for s in $SOLVERS; do
    case $s in
        direct|iterative|block) OPT="--linear-stokes $s";;
        bubbles) OPT="--linear-stokes iterative --stokes-element Bubbles";;
        amg) OPT="--linear-stokes block --stokes-block-file elmer/stokes_block_amg.sif.in";;
        *) echo "unknown solver $s"; exit 1;;
    esac
    python3 -u periodic_test/run_gl_ess.py run none --heat flux --stokes-only $OPT --tag ${TAG_PREFIX}$s --save-every 1 \
        --fbase ${FBASE:-10} ${CHUNKS:+--chunks $CHUNKS} $PHASE_ARGS ${RESUME:+--resume $RESUME} \
        ${START:+--start $START} ${SPINUP_TOL:+--spinup-tol $SPINUP_TOL} ${SPINUP_MAX:+--spinup-max $SPINUP_MAX} \
        ${SPINUP_ONLY:+--spinup-only} > ${L}_${TAG_PREFIX}$s.log 2>&1 &
    echo "$(date +%T) started $s (log ${L}_$s.log)"
done
FAIL=0
for p in $(jobs -p); do wait $p || FAIL=1; done
echo "$(date +%T) all runs done$([ $FAIL = 1 ] && echo ' (at least one failed: see its log)')"
python3 -u periodic_test/run_gl_ess.py solvers $(for s in $SOLVERS; do echo ${TAG_PREFIX}$s; done) || true
