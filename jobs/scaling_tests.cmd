#!/bin/bash
#SBATCH --output=./slurms/scaling_%j.out
#SBATCH --job-name=scaling
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --exclusive
#SBATCH --time=08:00:00
#
# Stokes solvers that might keep working on many partitions, at a given number of ranks (RANKS, default 32). Run
# it at two partition counts and compare the outer iterations per Stokes solve: a method that scales keeps them
# about the same when the ranks go up. Short runs (2 time steps, frozen grounding line) on the linear-bed test's
# mesh and start state (configs/test_gl_bed.toml, work dir work/test_gl_bed; GL_CONFIG / START for others):
#   R  GCR + ILU8 on the whole Stokes system (the current setup; the reference)
#   F  GCR + ILU0 (fails with a long shelf?)
#   S  block preconditioner, velocity block = Hypre BiCGStab + ParaSails (sparse approximate inverse: one operator
#      for the whole parallel matrix, not per partition)
#   T  the same with Hypre FSAI (factorised sparse approximate inverse)
#   U  block, velocity block = Hypre BiCGStab + BoomerAMG built on the uncoupled velocity components (HYPRE Block
#      Diagonal), the I tuning of jobs/amg_tests.cmd
#   C  GCR + ILU8 with the shelf CUT DIST km (default 30) downstream of the grounding line (ess run --shelf-cut:
#      the shelf beyond is passive for Stokes, the cut face carries the stress of a free-floating shelf)
#   D  the cut with GCR + ILU0      E  the cut with GCR + ILU1
# From the repository root, one job per partition count (each uses one node; the runs of a job share it):
#   sbatch --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,RANKS=32 jobs/scaling_tests.cmd
#   sbatch --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,RANKS=128 jobs/scaling_tests.cmd
# SET=CDE (any letters): only those runs (C, D, E also run R for their velocity comparison if R is not in SET).
# Summary at the end of slurms/scaling_<job>.out; per-run logs slurms/scaling_<job>_<letter>.log.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:?ELMER_MODULE not set}
export GL_CONFIG=${GL_CONFIG:-configs/test_gl_bed.toml}
WORK=work/test_gl_bed
START=${START:-$(ls $WORK/ref_iterative/spinup_chunks/points/state_*.npz 2>/dev/null | tail -1)}
[ -f "$START" ] || { echo "no start state (START=...)"; exit 1; }
RANKS=${RANKS:-32}
SET=${SET:-RFSTUCDE}
DIST=${DIST:-30}
case $SET in *[CDE]*) case $SET in *R*) ;; *) SET="R$SET";; esac;; esac
L=slurms/scaling_${SLURM_JOB_ID:-local}

bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
echo "Elmer: $ELMER_MODULE -> $(command -v ElmerSolver); $RANKS ranks per run; runs $SET; start $START"
MESH=$WORK/case/footprint
if [ ! -d $MESH/partitioning.$RANKS ]; then
    echo "$(date +%T) partitioning the footprint for $RANKS ranks (whole columns per rank: Elmer extrudes it)"
    ElmerGrid 2 2 $MESH -metiskway $RANKS > ${L}_partition.log 2>&1 || { cat ${L}_partition.log; exit 1; }
fi
[ -d $MESH/partitioning.$RANKS ] || { echo "no $MESH/partitioning.$RANKS after ElmerGrid"; exit 1; }
GL=$(python3 periodic_test/run_gl_ess.py glpos $START | tail -1)
CUT=$(python3 -c "print($GL + $DIST)")
echo "start grounding line $GL km; shelf cut (C, D, E) at $CUT km"
export ESS_NPROCS=$RANKS ESS_MPIRUN="mpirun -np {n} --bind-to none"
SLOTS=$(( ${SLURM_NTASKS:-128} / RANKS )); [ $SLOTS -ge 1 ] || SLOTS=1
common="run $GL_CONFIG --state $START --stokes-only --chunks 1 --steps 2 --no-remask --bed cold --gl frozen --linear-stokes block"

V=$WORK/scaling_files_${SLURM_JOB_ID:-local}; mkdir -p $V
mkGCR() {   # $1 file, $2 ILU level: GCR + ILU on the whole Stokes system, every iteration printed (counts)
    printf '%s\n' "  Linear System Solver = Iterative" "  Linear System Iterative Method = GCR" \
        "  Linear System GCR Restart = Integer 250" "  Linear System Max Iterations = 2000" \
        "  Linear System Preconditioning = ILU$2" "  Linear System Convergence Tolerance = 1.0e-8" \
        "  Linear System Abort Not Converged = Logical False" "  Linear System Residual Output = 1" > $1
}
mkBlock() {   # $1 file, $2 velocity-block preconditioner: block solver, velocity block Hypre BiCGStab + $2 capped
    grep -v "block 11: Linear System Iterative Method\|block 11: HYPRE Block Diagonal\|block 11: Linear System Preconditioning\|block 11: Linear System Max Iterations\|block 11: Linear System Convergence Tolerance" \
        elmer/stokes_block_amg.sif.in > $1
    printf '%s\n' "  block 11: Linear System Solver = Iterative" "  block 11: Linear System Iterative Method = BiCGStab" \
        "  block 11: Linear System Preconditioning = $2" "  block 11: Linear System Max Iterations = 50" \
        "  block 11: Linear System Convergence Tolerance = 1.0e-2" >> $1
}
for t in $(echo $SET | fold -w1); do
    case $t in
        R) mkGCR $V/R.sif.in 8;;  F) mkGCR $V/F.sif.in 0;;
        C) mkGCR $V/C.sif.in 8;;  D) mkGCR $V/D.sif.in 0;;  E) mkGCR $V/E.sif.in 1;;
        S) mkBlock $V/S.sif.in ParaSails;;
        T) mkBlock $V/T.sif.in FSAI;;
        U) mkBlock $V/U.sif.in BoomerAMG
           printf '%s\n' "  block 11: HYPRE Block Diagonal = Logical True" "  block 11: BoomerAMG Coarsen Type = Integer 10" \
               "  block 11: BoomerAMG Interpolation Type = Integer 6" "  block 11: BoomerAMG Strong Threshold = Real 0.5" >> $V/U.sif.in;;
        *) echo "unknown run $t"; exit 1;;
    esac
    echo "! ESS scaling $t ${SLURM_JOB_ID:-local}" >> $V/$t.sif.in
done

running=0
for t in $(echo $SET | fold -w1); do
    extra=""; case $t in C|D|E) extra="--shelf-cut $CUT";; esac
    if [ $running -ge $SLOTS ]; then wait -n || true; running=$((running - 1)); fi
    echo "$(date +%T) start $t"
    python3 -u -m ess $common --out $WORK/scaling_${RANKS}_$t.npz --stokes-block-file $PWD/$V/$t.sif.in $extra \
        > ${L}_$t.log 2>&1 &
    running=$((running + 1))
done
wait || true

echo
echo "===== $RANKS ranks, $(date +%T) ====="
for t in $(echo $SET | fold -w1); do
    case $t in R) what="GCR + ILU8, whole shelf (reference)";; F) what="GCR + ILU0, whole shelf";;
               S) what="block, velocity block BiCGStab + ParaSails";; T) what="block, velocity block BiCGStab + FSAI";;
               U) what="block, velocity block BiCGStab + BoomerAMG on uncoupled components";;
               C) what="GCR + ILU8, shelf cut at $CUT km";; D) what="GCR + ILU0, shelf cut";;
               E) what="GCR + ILU1, shelf cut";; esac
    if grep -q "wrote $WORK/scaling_${RANKS}_$t.npz" ${L}_$t.log; then
        echo "$t ($what): RAN   $(grep -o "Elmer [0-9]* s for .*" ${L}_$t.log | tail -1)"
        f=$(grep -l "ESS scaling $t ${SLURM_JOB_ID:-local}\$" $WORK/runs/*/run.sif 2>/dev/null | tail -1)
        if [ -n "$f" ]; then
            el=$(dirname $f)/elmer.log
            # outer iterations per Stokes solve (the counter restarts at every solve) and the final residual
            grep "gcr:" $el | awk '$2 < prev {n++; printf "      solve %d: %4d iterations, residual %s\n", n, prev, res}
                                  {prev=$2; res=$3} END {n++; printf "      solve %d: %4d iterations, residual %s\n", n, prev, res}'
            hc=$(grep -c "did not converge" $el || true); [ "$hc" -gt 0 ] && echo "      Hypre 'did not converge' messages: $hc"
        fi
        case $t in C|D|E) [ -f $WORK/scaling_${RANKS}_R.npz ] && \
            python3 periodic_test/run_gl_ess.py cutcheck $WORK/scaling_${RANKS}_R.npz $WORK/scaling_${RANKS}_$t.npz $CUT $GL | tail -1;; esac
    else
        log=$(grep -o "see [^ ]*elmer.log" ${L}_$t.log | tail -1 | cut -c5-)
        echo "$t ($what): FAILED   $log"
        [ -n "$log" ] && grep -m3 -i "double free\|corruption\|error\|signal\|fatal" $log | sed 's/^/      /'
    fi
done
