#!/bin/bash
# Stokes solver test with a moving grounding line (configs/test_gl_solvers.toml): the marine flowline, the contact
# grounding line, spin-up to steady, retreat (accumulation x 0.3), advance (x 3); Stokes + free surface only. The
# same run once per Stokes linear solver, each on its own node (exclusive, 32 ranks), so the times compare:
#   direct     MUMPS
#   iterative  GCR + ILU8
#   block      block preconditioning, elmer/stokes_block.sif.in (MLB block2_idrs: GCR outside, IDR(s) + ILU inside)
#   amg        block preconditioning with BoomerAMG on the velocity block (elmer/stokes_block_amg.sif.in, Hypre)
# Block preconditioning with the normal-tangential bed needs an Elmer with the 2026-10-14 fix (ElmerCSC/elmerfem
# 30a191d): ELMER_MODULE names it. Run on the login node from the repository root:
#   ELMER_MODULE=elmerfem/<new version> bash jobs/gl_solvers.sh
#   SOLVERS="direct iterative block amg" ELMER_MODULE=... bash jobs/gl_solvers.sh     # also BoomerAMG
#   NOPREP=1 ...           the mesh and start state already there (work/test_gl_solvers)
#   CHUNKS=2 ...           a short trial (2 chunks = 200 yr per phase)
#   EXCLUSIVE=0 ...        no whole node per run (they start sooner when the cluster is full; noisier timings)
# Results:  sbatch --ntasks=1 --time=00:30:00 --export=ALL,GL_CONFIG=configs/test_gl_solvers.toml \
#               periodic_test/run_gl_ess.cmd solvers
# (or on a laptop: GL_CONFIG=configs/test_gl_solvers.toml python periodic_test/run_gl_ess.py solvers)
set -e
cd "$(dirname "$0")/.."
: "${ELMER_MODULE:?set ELMER_MODULE to the Elmer module with the block-preconditioner fix, e.g. ELMER_MODULE=elmerfem/2026-10-xx}"
export ELMER_MODULE GL_CONFIG=configs/test_gl_solvers.toml
FROM=${FROM:-work/marine_flowline/run_cold_p268.15.npz}
[ -f "$FROM" ] || { echo "start state $FROM not found (FROM=...)"; exit 1; }
SB="sbatch --parsable --export=ALL"
# EXCLUSIVE=0: the runs share nodes (start sooner; noisier timings)
[ "${EXCLUSIVE:-1}" = 0 ] && EXCLUSIVE_FLAG="" || EXCLUSIVE_FLAG="--exclusive"
B=$($SB --ntasks=1 --time=00:20:00 periodic_test/run_gl_ess.cmd build)
DEP="--dependency=afterok:$B"
echo "build ($ELMER_MODULE): job $B"
if [ -z "$NOPREP" ]; then
    P=$($SB $DEP periodic_test/run_gl_ess.cmd prep --from "$FROM")
    DEP="--dependency=afterok:$P"
    echo "prep (mesh, start state from $FROM): job $P"
fi
for s in ${SOLVERS:-direct iterative block}; do
    case $s in
        direct|iterative|block) OPT="--linear-stokes $s";;
        bubbles) OPT="--linear-stokes iterative --stokes-element Bubbles";;
        amg) OPT="--linear-stokes block --stokes-block-file elmer/stokes_block_amg.sif.in";;
        *) echo "unknown solver $s"; exit 1;;
    esac
    J=$($SB ${EXCLUSIVE_FLAG} --job-name=glsolv_$s $DEP periodic_test/run_gl_ess.cmd run none --heat flux --stokes-only \
          $OPT --tag $s --save-every 1 --fbase ${FBASE:-10} ${CHUNKS:+--chunks $CHUNKS})
    echo "$s: job $J (work/test_gl_solvers/$s)"
done
echo "logs: slurms/pgless_<job>.out (the job name is glsolv_<solver>)"
echo "compare: sbatch --ntasks=1 --time=00:30:00 --export=ALL,GL_CONFIG=$GL_CONFIG periodic_test/run_gl_ess.cmd solvers"
