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
#   C  GCR + ILU8 with the shelf CUT DIST km (default 30) downstream of the grounding line: a shorter mesh (same
#      nodes upstream, [case] front_km) whose calving front, with sea pressure, is at the cut. For an unbuttressed
#      shelf that front transmits the same force to the grounded ice as the whole shelf (FS-SSA coupling with the
#      SSA shelf solved exactly). The first job makes the mesh (work/test_gl_bed_cut<km>) and carries the start
#      state onto it (ess transfer).
#   D  the cut with GCR + ILU0      E  the cut with GCR + ILU1
# From the repository root, one job per partition count (each uses one node; the runs of a job share it):
#   sbatch --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,RANKS=32 jobs/scaling_tests.cmd
#   sbatch --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,RANKS=128 jobs/scaling_tests.cmd
# 3D sheet: GL_CONFIG=configs/test_periodic_gl_50km.toml (its work dir's start.npz, START= for another state);
# periodic sides are partitioned in boxes RANKS/4 x 4 (as the production 64 x 4). More ranks than a node:
#   sbatch --nodes=2 --ntasks=256 --export=ALL,...,GL_CONFIG=configs/test_periodic_gl_50km.toml,RANKS=256 ...
# With y_elements > 1 the cut's start state is laterally averaged (ess transfer works on columns along x), so
# the whole-shelf runs (R, F, S, T, U) start from the same average, carried onto the full mesh.
# SET=CDE (any letters): only those runs (C, D, E also run R for their velocity comparison if R is not in SET).
# Run the two jobs one after the other or at the same time (the mesh steps are locked).
# Summary at the end of slurms/scaling_<job>.out; per-run logs slurms/scaling_<job>_<letter>.log.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:?ELMER_MODULE not set}
export GL_CONFIG=${GL_CONFIG:-configs/test_gl_bed.toml}
WORK=$(python3 -c "import sys, os; sys.path.insert(0, '.'); from ess.config import load_config; print(os.path.relpath(load_config('$GL_CONFIG')['_work']))" | tail -1)
CNAME=$(basename $GL_CONFIG .toml)
if [ -z "$START" ]; then
    if [ "$CNAME" = test_gl_bed ]; then START=$(ls $WORK/ref_iterative/spinup_chunks/points/state_*.npz 2>/dev/null | tail -1)
    else      # the newest saved chunk of any run in the work dir (spun up), else <work dir>/start.npz
        START=$(ls -t $WORK/*/*_chunks/points/state_*.npz 2>/dev/null | head -1)
        [ -n "$START" ] || START=$WORK/start.npz
    fi
fi
[ -f "$START" ] || { echo "no start state in $WORK (START=...): $(ls $WORK 2>&1 | head -20 | tr '\n' ' ')"; exit 1; }
RANKS=${RANKS:-32}
SET=${SET:-RFSTUCDE}
DIST=${DIST:-30}
case $SET in *[CDE]*) case $SET in *R*) ;; *) SET="R$SET";; esac;; esac
L=slurms/scaling_${SLURM_JOB_ID:-local}

bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
echo "Elmer: $ELMER_MODULE -> $(command -v ElmerSolver); $RANKS ranks per run; runs $SET; start $START"
partition() {   # $1 case dir (with footprint.msh), $2 config: add footprint/partitioning.$RANKS, converted from
    # footprint.msh exactly as `ess setup` does (same node numbering; the mesh dir only holds partitioned copies).
    # The 2D footprint: Elmer extrudes it, so every rank gets whole columns.
    [ -d $1/footprint/partitioning.$RANKS ] && return 0
    echo "$(date +%T) partitioning $1/footprint for $RANKS ranks"
    local per="" part="-metiskway $RANKS"
    if grep -q "^periodic_sides *= *true" $2; then per="-periodic 0 1 0"; part="-partition $PX $PY 1"; fi
    rm -rf $1/fp_part$RANKS
    ( cd $1 && ElmerGrid 14 2 footprint.msh -autoclean -out fp_part$RANKS $per $part ) \
        > ${L}_partition.log 2>&1 || { cat ${L}_partition.log; return 1; }
    [ -d $1/fp_part$RANKS/partitioning.$RANKS ] || { echo "ElmerGrid made no partitioning.$RANKS:"; tail -20 ${L}_partition.log; return 1; }
    mv $1/fp_part$RANKS/partitioning.$RANKS $1/footprint/ && rm -rf $1/fp_part$RANKS
}
PY=1; [ $((RANKS % 4)) = 0 ] && PY=4; PX=$((RANKS / PY))     # box partitions for periodic sides (as 64 x 4)
YEL=$(grep -o "^y_elements *= *[0-9]*" $GL_CONFIG | grep -o "[0-9]*$" || echo 1)
( flock 9; partition $WORK/case $GL_CONFIG ) 9> $WORK/case/.lock
GL=$(python3 periodic_test/run_gl_ess.py glpos $START | tail -1)
CUT=$(python3 -c "print($GL + $DIST)")
echo "start grounding line $GL km; shelf cut (C, D, E) at $CUT km"
export ESS_NPROCS=$RANKS ESS_MPIRUN="mpirun -np {n} --bind-to none"
case $SET in *[CDE]*)
    CW=work/${CNAME}_c$(python3 -c "print(round($CUT))")     # short: Elmer cuts mesh paths at 128 characters
    CUTCFG=$CW/cut.toml
    mkdir -p $CW
    (
    flock 9
    if [ ! -f $CW/start.npz ]; then
        echo "$(date +%T) the cut mesh: front at $CUT km ($CW, log ${L}_cutprep.log)"
        python3 - "$GL_CONFIG" "$CUTCFG" "$CW" "$CUT" "$PX" "$PY" <<'PYEND'
import re, sys
src, dst, work, cut, px, py = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6]
s = open(src).read()
s, n1 = re.subn(r'^work_dir = .*$', f'work_dir = "{work}"', s, flags=re.M)
s, n2 = re.subn(r'^\[case\]\s*$', f'[case]\nfront_km = {cut}', s, flags=re.M)
if n1 != 1 or n2 != 1:
    sys.exit("cut config: work_dir / [case] lines not found in " + src)
# a fine band given directly ends just before the front; box partitions for this rank count
s = re.sub(r'^fine_band_km = \[\s*([0-9.]+)\s*,\s*[0-9.]+\s*\]',
           lambda m: f'fine_band_km = [{m.group(1)}, {cut - 2.5}]', s, flags=re.M)
s = re.sub(r'^partition = \[.*?\]', f'partition = [{px}, {py}]', s, flags=re.M)
# ess init solves Stokes once (not used): ILU0, as ILU8 on few large 3D partitions overflows Elmer's integers
s, n3 = re.subn(r'^stokes_preconditioner = .*$', 'stokes_preconditioner = "ILU0"', s, flags=re.M)
if n3 == 0:
    s = s.replace("[elmer]\n", '[elmer]\nstokes_preconditioner = "ILU0"\n', 1)
open(dst, "w").write(s)
PYEND
        { python3 -u -m ess setup $CUTCFG && python3 -u -m ess init $CUTCFG --out $CW/state0.npz && \
          python3 -u -m ess transfer $CUTCFG --state $CW/state0.npz --from $START --out $CW/start.npz; } \
            > ${L}_cutprep.log 2>&1 || { echo "cut mesh preparation failed:"; tail -30 ${L}_cutprep.log; exit 1; }
    fi
    partition $CW/case $CUTCFG
    ) 9> $CW.lock || exit 1
    echo "cut mesh $CW, start $CW/start.npz"
;; esac
RSTART=$START
if [ "${YEL:-1}" -gt 1 ]; then          # 3D: the whole-shelf runs from the same laterally averaged state as the cut
    RSTART=$WORK/scaling_start_avg.npz
    ( flock 9
      [ -f $RSTART ] || python3 -u -m ess transfer $GL_CONFIG --state $START --from $START --out $RSTART \
          > ${L}_avgprep.log 2>&1 || { echo "averaged start failed:"; tail -30 ${L}_avgprep.log; exit 1; }
    ) 9> $WORK/case/.lock || exit 1
    echo "whole-shelf runs start from $RSTART (laterally averaged, as the cut's)"
fi
SLOTS=${SLOTS:-$(( ${SLURM_NTASKS:-128} / RANKS ))}; [ $SLOTS -ge 1 ] || SLOTS=1   # runs at a time (SLOTS=1: one by one)
opts="--stokes-only --chunks 1 --steps ${STEPS:-2} --no-remask --bed cold --gl frozen --linear-stokes block"
# STOKES_ELEMENT=Stabilized|Bubbles|P2/P1: the element for every run (default the config's). Low ILU needs
# Stabilized? (the flowline tests, where ILU0/ILU1 converged, were Stabilized; the 50 km config is Bubbles, where
# they did not reduce the residual at all)
[ -n "$STOKES_ELEMENT" ] && opts="$opts --stokes-element $STOKES_ELEMENT"
CFG_ELEMENT=$(grep -o '^stokes_element *= *"[^"]*"' $GL_CONFIG | cut -d'"' -f2)
echo "Stokes element: ${STOKES_ELEMENT:-$CFG_ELEMENT, from the config}"

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
    c="run $GL_CONFIG --state $RSTART $opts"; o=$WORK/scaling_${RANKS}_$t.npz
    case $t in C|D|E) c="run $CUTCFG --state $CW/start.npz $opts"; o=$CW/scaling_${RANKS}_$t.npz;; esac
    if [ $running -ge $SLOTS ]; then wait -n || true; running=$((running - 1)); fi
    echo "$(date +%T) start $t"
    python3 -u -m ess $c --out $o --stokes-block-file $PWD/$V/$t.sif.in > ${L}_$t.log 2>&1 &
    running=$((running + 1))
done
wait || true

echo
echo "===== $RANKS ranks, $(date +%T) ====="
for t in $(echo $SET | fold -w1); do
    case $t in R) what="GCR + ILU8, whole shelf (reference)";; F) what="GCR + ILU0, whole shelf";;
               S) what="block, velocity block BiCGStab + ParaSails";; T) what="block, velocity block BiCGStab + FSAI";;
               U) what="block, velocity block BiCGStab + BoomerAMG on uncoupled components";;
               C) what="GCR + ILU8, mesh ends at $CUT km";; D) what="GCR + ILU0, shelf cut";;
               E) what="GCR + ILU1, shelf cut";; esac
    W=$WORK; case $t in C|D|E) W=$CW;; esac
    if grep -q "wrote $W/scaling_${RANKS}_$t.npz" ${L}_$t.log; then
        echo "$t ($what): RAN   $(grep -o "Elmer [0-9]* s for .*" ${L}_$t.log | tail -1)"
        f=$(grep -l "ESS scaling $t ${SLURM_JOB_ID:-local}\$" $W/runs/*/run.sif 2>/dev/null | tail -1)
        if [ -n "$f" ]; then
            el=$(dirname $f)/elmer.log
            # outer iterations per Stokes solve (the counter restarts at every solve) and the final residual
            grep "gcr:" $el | awk '$2 < prev {n++; printf "      solve %d: %4d iterations, residual %s\n", n, prev, res}
                                  {prev=$2; res=$3} END {n++; printf "      solve %d: %4d iterations, residual %s\n", n, prev, res}'
            hc=$(grep -c "did not converge" $el || true); [ "$hc" -gt 0 ] && echo "      Hypre 'did not converge' messages: $hc"
        fi
        case $t in C|D|E) [ -f $WORK/scaling_${RANKS}_R.npz ] && \
            python3 periodic_test/run_gl_ess.py cutcheck $WORK/scaling_${RANKS}_R.npz $CW/scaling_${RANKS}_$t.npz $CUT $GL | tail -1;; esac
    else
        log=$(grep -o "see [^ ]*elmer.log" ${L}_$t.log | tail -1 | cut -c5-)
        echo "$t ($what): FAILED   $log"
        [ -n "$log" ] && grep -m3 -i "double free\|corruption\|error\|signal\|fatal" $log | sed 's/^/      /'
    fi
done
