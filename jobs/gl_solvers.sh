#!/bin/bash
# Stokes solver test with a moving grounding line (configs/test_gl_solvers.toml): the marine flowline, the contact
# grounding line, spin-up to steady, retreat (accumulation x 0.3), advance (x 3); Stokes + free surface only.
# All solvers run AT THE SAME TIME in ONE job on ONE node (jobs/gl_solvers_node.cmd), RANKS (default 32) ranks
# each, so they start together and share the same conditions:
#   direct     MUMPS
#   iterative  GCR + ILU8
#   block      block preconditioning, elmer/stokes_block.sif.in (MLB block2_idrs: GCR outside, IDR(s) + ILU inside)
#   bubbles    GCR + ILU8 with the bubbles element instead of the stabilized one
#   amg        block preconditioning with BoomerAMG on the velocity block (elmer/stokes_block_amg.sif.in, Hypre)
# Block preconditioning with the normal-tangential bed needs an Elmer with the 2026-10-14 fix (ElmerCSC/elmerfem
# 30a191d): ELMER_MODULE names it. Run on the login node from the repository root:
#   ELMER_MODULE=elmerfem/2026-10-28 bash jobs/gl_solvers.sh                        # direct iterative block
#   SOLVERS="direct iterative block bubbles" ELMER_MODULE=... bash jobs/gl_solvers.sh
#   CHUNKS=2 ...           a short trial (2 chunks = 200 yr per phase)
#   PHASES="retreat=50,1,0 advance=50,1,3" ...   phases NAME=CHUNKS,FRICTION,ACCUMULATION (the defaults, 10 chunks of
#                          accumulation x 0.3 / x 3, are too weak to move the grounding line: response time ~30 kyr)
#   RESUME=retreat ...     keep the finished spin-ups, rerun retreat and advance (e.g. with new PHASES)
#   GL_CONFIG=configs/test_gl_solvers_fine.toml FROM=<state> ...   another config (its own work dir) and start state
# The linear-bed test (configs/test_gl_bed.toml): 1. one steady spin-up, 2. all solvers from it:
#   GL_CONFIG=configs/test_gl_bed.toml FROM=none SOLVERS=iterative TAG_PREFIX=ref_ SPINUP_ONLY=1 \
#     SPINUP_TOL=1e-4 SPINUP_MAX=600 ELMER_MODULE=... bash jobs/gl_solvers.sh
#   GL_CONFIG=configs/test_gl_bed.toml START=work/test_gl_bed/ref_iterative/spinup.npz \
#     SOLVERS="direct iterative block bubbles" PHASES="spinup=1,1,1 retreat=100,1,0.7 advance=100,1,1.5" \
#     ELMER_MODULE=... bash jobs/gl_solvers.sh
# SOLVERS x RANKS must fit the node (128 cores). The mesh and start state are made in the job if missing
# (work/test_gl_solvers/start.npz). Results: at the end of slurms/glsolvers_<job>.out; again with
#   GL_CONFIG=configs/test_gl_solvers.toml python3 periodic_test/run_gl_ess.py solvers
set -e
cd "$(dirname "$0")/.."
: "${ELMER_MODULE:?set ELMER_MODULE to the Elmer module with the block-preconditioner fix, e.g. ELMER_MODULE=elmerfem/2026-10-28}"
export GL_CONFIG=${GL_CONFIG:-configs/test_gl_solvers.toml}
export ELMER_MODULE PHASES RESUME SOLVERS=${SOLVERS:-direct iterative block} RANKS=${RANKS:-32}
FROM=${FROM:-work/marine_flowline/run_cold_p268.15.npz}
[ "$FROM" = none ] || [ -f "$FROM" ] || { echo "start state $FROM not found (FROM=..., or none: the config's own)"; exit 1; }
export START TAG_PREFIX SPINUP_TOL SPINUP_MAX SPINUP_ONLY
export FROM
N=$(( $(wc -w <<<"$SOLVERS") * RANKS ))
[ $N -le 128 ] || { echo "$SOLVERS x $RANKS ranks = $N > 128 cores of a node: fewer SOLVERS or RANKS"; exit 1; }
J=$(sbatch --parsable --export=ALL --ntasks=$N jobs/gl_solvers_node.cmd)
echo "job $J: $SOLVERS at the same time on one node, $RANKS ranks each ($N tasks, $ELMER_MODULE, $GL_CONFIG)"
echo "logs: slurms/glsolvers_$J.out, slurms/glsolvers_${J}_<solver>.log; results in work/test_gl_solvers/<solver>"
