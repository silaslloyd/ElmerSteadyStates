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
# SET=G: block 11 as in Elmer's own test fem/tests/mgdyn_airgap2 (Hypre BiCGStab preconditioned by BoomerAMG,
# instead of BoomerAMG as the solver with HYPRE Block Diagonal as in MLB), frozen GL, rotated bed
#                            sbatch --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,SET=G jobs/amg_tests.cmd
# SET=RHIJ (4 runs: sbatch --ntasks=128): tuning G against plain GCR + ILU8, all frozen GL, 2 steps; compare the Elmer
# time each prints. R GCR + ILU8 (the reference)   H G with the inner solve capped (30 iterations, tolerance 1e-2)
#   I H + HMIS coarsening, extended interpolation, strong threshold 0.5   J I + strong threshold 0.25, 2 sweeps
#                            sbatch --ntasks=128 --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,SET=RHIJ jobs/amg_tests.cmd
# SET=RKLM (sbatch --ntasks=128): I with a pressure (Schur complement) preconditioner for block 22 (I alone stalls at
# ~2e-4: the outer GCR hits 200 iterations). R GCR + ILU8 (reference)   K I + Create Schur Matrix Approximation
# (SIMPLE: -B diag(A)^-1 B^T)   L I + pressure mass matrix x -1/viscosity   M I + pressure mass matrix x +1/viscosity
#                            sbatch --ntasks=128 --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,SET=RKLM jobs/amg_tests.cmd
# SET=NOPQ (sbatch --ntasks=128): solvers that couple the partitions (ILU8 degrades with many partitions)
#   N  I with the bubbles element (no stabilisation term in the pressure block; the -1/viscosity pressure
#      preconditioner Stokes-Vec builds is then the textbook one)
#   O  I with the velocity block solved tightly (1e-6, up to 300 iterations): is the outer stall from the inexact
#      inner solve?
#   P  Hypre FlexGMRES + Hypre ILU type 30 (restricted additive Schwarz ILU(0), overlapping partitions), whole system
#   Q  Hypre FlexGMRES + Hypre ILU type 10 (GMRES-ILU(0): partition coupling through a Schur complement)
#                            sbatch --ntasks=128 --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,SET=NOPQ jobs/amg_tests.cmd
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
mkG() {   # G: block 11 as in Elmer's mgdyn_airgap2 (Hypre BiCGStab with BoomerAMG as preconditioner)
    grep -v "block 11: Linear System Iterative Method\|block 11: HYPRE Block Diagonal\|block 11: Linear System Preconditioning" \
        elmer/stokes_block_amg.sif.in > $1
    printf '%s\n' "  block 11: Linear System Solver = Iterative" "  block 11: Linear System Iterative Method = BiCGStab" \
        "  block 11: Linear System Preconditioning = BoomerAMG" >> $1
}
mkI() {   # I: G, inner solve capped, HMIS coarsening, extended interpolation, strong threshold 0.5
    mkG $1.G
    grep -v "block 11: Linear System Max Iterations\|block 11: Linear System Convergence Tolerance" $1.G > $1
    printf '%s\n' "  block 11: Linear System Max Iterations = 30" "  block 11: Linear System Convergence Tolerance = 1.0e-2" \
        "  block 11: BoomerAMG Coarsen Type = Integer 10" "  block 11: BoomerAMG Interpolation Type = Integer 6" \
        "  block 11: BoomerAMG Strong Threshold = Real 0.5" >> $1
}
mkHILU() {   # Hypre FlexGMRES + Hypre ILU of type $2 on the whole Stokes system
    printf '%s\n' "  Linear System Solver = Iterative" "  Linear System Use Hypre = Logical True" \
        "  Linear System Iterative Method = FlexGMRES" "  Linear System Preconditioning = ILU$2" \
        "  Hypre GMRES Dimension = Integer 200" "  Linear System Max Iterations = 2000" \
        "  Linear System Convergence Tolerance = 1.0e-8" "  Linear System Abort Not Converged = Logical False" \
        "  Linear System Residual Output = 50" > $1
}
if [ "$SET" = NOPQ ]; then
    V=$WORK/amgtest_files; mkdir -p $V
    mkI $V/N.sif.in; echo "! ESS amgtest N" >> $V/N.sif.in
    mkI $V/O.sif.in.tmp
    grep -v "block 11: Linear System Max Iterations\|block 11: Linear System Convergence Tolerance" $V/O.sif.in.tmp > $V/O.sif.in
    printf '%s\n' "  block 11: Linear System Max Iterations = 300" "  block 11: Linear System Convergence Tolerance = 1.0e-6" \
        "! ESS amgtest O" >> $V/O.sif.in
    mkHILU $V/P.sif.in 30; echo "! ESS amgtest P" >> $V/P.sif.in
    mkHILU $V/Q.sif.in 10; echo "! ESS amgtest Q" >> $V/Q.sif.in
    python3 -u -m ess $common --out $WORK/amgtest_N.npz --gl frozen --stokes-element Bubbles \
        --stokes-block-file $PWD/$V/N.sif.in > ${L}_N.log 2>&1 &
    for t in O P Q; do
        python3 -u -m ess $common --out $WORK/amgtest_$t.npz --gl frozen --stokes-block-file $PWD/$V/$t.sif.in \
            > ${L}_$t.log 2>&1 &
    done
elif [ "$SET" = RKLM ]; then
    V=$WORK/amgtest_files; mkdir -p $V
    IV=$(python3 -c "from ess.case import eval_lua_parameters as e; print(repr(1 / e('case/parameters.lua')['viscosity']))" 2>/dev/null | tail -1)
    for t in K L M; do mkI $V/$t.sif.in; echo "! ESS amgtest $t" >> $V/$t.sif.in; done
    echo "  Create Schur Matrix Approximation = Logical True" >> $V/K.sif.in
    echo "  Prec Matrix Density 2 = Real -$IV" >> $V/L.sif.in
    echo "  Prec Matrix Density 2 = Real $IV" >> $V/M.sif.in
    python3 -u -m ess ${common% --linear-stokes block} --linear-stokes iterative --out $WORK/amgtest_R.npz --gl frozen \
        > ${L}_R.log 2>&1 &
    for t in K L M; do
        python3 -u -m ess $common --out $WORK/amgtest_$t.npz --gl frozen --stokes-block-file $PWD/$V/$t.sif.in \
            > ${L}_$t.log 2>&1 &
    done
elif [ "$SET" = RHIJ ]; then
    V=$WORK/amgtest_files; mkdir -p $V
    mkG $V/G.sif.in
    grep -v "block 11: Linear System Max Iterations\|block 11: Linear System Convergence Tolerance" $V/G.sif.in > $V/H.sif.in
    printf '%s\n' "  block 11: Linear System Max Iterations = 30" "  block 11: Linear System Convergence Tolerance = 1.0e-2" >> $V/H.sif.in
    cp $V/H.sif.in $V/I.sif.in
    printf '%s\n' "  block 11: BoomerAMG Coarsen Type = Integer 10" "  block 11: BoomerAMG Interpolation Type = Integer 6" \
        "  block 11: BoomerAMG Strong Threshold = Real 0.5" >> $V/I.sif.in
    grep -v "block 11: BoomerAMG Strong Threshold" $V/I.sif.in > $V/J.sif.in
    printf '%s\n' "  block 11: BoomerAMG Strong Threshold = Real 0.25" "  block 11: BoomerAMG Num Sweeps = Integer 2" >> $V/J.sif.in
    python3 -u -m ess ${common% --linear-stokes block} --linear-stokes iterative --out $WORK/amgtest_R.npz --gl frozen \
        > ${L}_R.log 2>&1 &
    for t in H I J; do
        python3 -u -m ess $common --out $WORK/amgtest_$t.npz --gl frozen --stokes-block-file $PWD/$V/$t.sif.in \
            > ${L}_$t.log 2>&1 &
    done
elif [ "$SET" = G ]; then
    V=$WORK/amgtest_files; mkdir -p $V
    mkG $V/G.sif.in
    python3 -u -m ess $common --out $WORK/amgtest_G.npz --gl frozen --stokes-block-file $PWD/$V/G.sif.in \
        > ${L}_G.log 2>&1 &
elif [ "$SET" = DEF ]; then
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
               E) what="block + BoomerAMG without BoomerAMG Num Functions";; F) what="block + BoomerAMG without both";;
               G) what="block, block 11 = Hypre BiCGStab + BoomerAMG preconditioner (as Elmer's mgdyn_airgap2)";;
               R) what="GCR + ILU8, the reference";; H) what="G, inner solve capped at 30 iterations / 1e-2";;
               I) what="H + HMIS coarsening, extended interpolation, strong threshold 0.5";;
               J) what="I + strong threshold 0.25, 2 sweeps";;
               K) what="I + Schur approximation -B diag(A)^-1 B^T for the pressure";;
               L) what="I + pressure mass matrix x -1/viscosity";; M) what="I + pressure mass matrix x +1/viscosity";;
               N) what="I with the bubbles element";; O) what="I with the velocity block solved to 1e-6";;
               P) what="Hypre FlexGMRES + RAS-ILU(0) (overlapping partitions)";;
               Q) what="Hypre FlexGMRES + GMRES-ILU(0) (Schur coupling of partitions)";; esac
    if grep -q "^.*wrote $WORK/amgtest_$t.npz" ${L}_$t.log; then
        echo "$t ($what): RAN   $(grep -o "Elmer [0-9]* s for .*" ${L}_$t.log | tail -1)"
        f=$(grep -l "ESS amgtest $t\$" $WORK/runs/*/run.sif 2>/dev/null | tail -1)
        [ -n "$f" ] && echo "      Hypre 'did not converge' messages: $(grep -c "did not converge" $(dirname $f)/elmer.log)"
        [ -n "$f" ] && grep -q "gcr:" $(dirname $f)/elmer.log && echo "      last outer GCR line of each Stokes solve:" && \
            grep "gcr:" $(dirname $f)/elmer.log | awk '$2 < prev {print last} {prev=$2; last=$0} END {print last}' | sed 's/^/      /'
    else
        log=$(grep -o "see [^ ]*elmer.log" ${L}_$t.log | tail -1 | cut -c5-)
        echo "$t ($what): FAILED   $log"
        [ -n "$log" ] && grep -m3 -i "double free\|corruption\|error\|signal" $log | sed 's/^/      /'
    fi
done
