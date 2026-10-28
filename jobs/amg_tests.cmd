#!/bin/bash
#SBATCH --output=./slurms/amgtests_%j.out
#SBATCH --job-name=amgtests
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --exclusive
#SBATCH --time=01:00:00
#
# Where does the BoomerAMG block preconditioner crash ("double free or corruption" in the first Stokes solve)?
# Three short runs (2 time steps each, 32 ranks each, at the same time) on the linear-bed test's existing mesh and
# start state (configs/test_gl_bed.toml, work dir work/test_gl_bed; GL_CONFIG / START to use another):
#   A  amg_frozen  block + BoomerAMG, grounding line frozen (no contact), the bed rotated (normal-tangential)
#   B  amg_cart    block + BoomerAMG, grounding line frozen, the bed in x/y/z components (no rotation)
#   C  hypre       Hypre BiCGStab + BoomerAMG on the whole Stokes system (no block preconditioning), with contact
# Reading the result:  A crashes, B runs   -> the block/Hypre path with the rotated sliding bed
#                      A and B crash, C runs -> the block solver's Hypre path in general
#                      C crashes too        -> Elmer's Hypre interface in this build (e.g. a Hypre version mismatch)
#                      A runs               -> the contact grounding line (the crashing run had contact)
# Second set (SET=DEF): which keyword of the BoomerAMG block setup triggers the crash (all frozen GL, rotated bed):
#   D  without "HYPRE Block Diagonal"   E  without "BoomerAMG Num Functions"   F  without both
# From the repository root:  sbatch --export=ALL,ELMER_MODULE=elmerfem/2026-10-28 jobs/amg_tests.cmd
#                            sbatch --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,SET=DEF jobs/amg_tests.cmd
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:?ELMER_MODULE not set}
CONFIG=${GL_CONFIG:-configs/test_gl_bed.toml}
WORK=work/test_gl_bed
START=${START:-$(ls $WORK/ref_iterative/spinup_chunks/points/state_*.npz 2>/dev/null | tail -1)}
[ -f "$START" ] || { echo "no start state (START=...)"; exit 1; }
L=slurms/amgtests_${SLURM_JOB_ID:-local}

bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
echo "Elmer: $ELMER_MODULE -> $(command -v ElmerSolver); ldd: $(ldd $(command -v ElmerSolver) | grep -i hypre | tr -s ' ')"
export ESS_NPROCS=32 ESS_MPIRUN="mpirun -np {n} --bind-to none"
common="run $CONFIG --state $START --stokes-only --chunks 1 --steps 2 --no-remask --bed cold --linear-stokes block"
SET=${SET:-ABC}
if [ "$SET" = DEF ]; then
    V=$WORK/amgtest_files; mkdir -p $V
    grep -v "HYPRE Block Diagonal" elmer/stokes_block_amg.sif.in > $V/D.sif.in
    grep -v "BoomerAMG Num Functions" elmer/stokes_block_amg.sif.in > $V/E.sif.in
    grep -v "HYPRE Block Diagonal\|BoomerAMG Num Functions" elmer/stokes_block_amg.sif.in > $V/F.sif.in
    for t in D E F; do
        python3 -u -m ess $common --out $WORK/amgtest_$t.npz --gl frozen --stokes-block-file $PWD/$V/$t.sif.in \
            > ${L}_$t.log 2>&1 &
    done
else
    python3 -u -m ess $common --out $WORK/amgtest_A.npz --gl frozen \
        --stokes-block-file elmer/stokes_block_amg.sif.in > ${L}_A.log 2>&1 &
    python3 -u -m ess $common --out $WORK/amgtest_B.npz --gl frozen --cartesian-bed \
        --stokes-block-file elmer/stokes_block_amg.sif.in > ${L}_B.log 2>&1 &
    python3 -u -m ess $common --out $WORK/amgtest_C.npz --gl contact \
        --stokes-block-file elmer/stokes_hypre.sif.in > ${L}_C.log 2>&1 &
fi
wait || true
echo
for t in $(echo $SET | fold -w1); do
    case $t in A) what="block + BoomerAMG, frozen GL, rotated bed";; B) what="block + BoomerAMG, frozen GL, x/y/z bed";;
               C) what="plain Hypre BoomerAMG, contact";; D) what="block + BoomerAMG without HYPRE Block Diagonal";;
               E) what="block + BoomerAMG without BoomerAMG Num Functions";; F) what="block + BoomerAMG without both";; esac
    if grep -q "^.*wrote $WORK/amgtest_$t.npz" ${L}_$t.log; then
        echo "$t ($what): RAN"
    else
        log=$(grep -o "see [^ ]*elmer.log" ${L}_$t.log | tail -1 | cut -c5-)
        echo "$t ($what): FAILED   $log"
        [ -n "$log" ] && grep -m3 -i "double free\|corruption\|error\|signal" $log | sed 's/^/      /'
    fi
done
