#!/bin/bash
#SBATCH --output=./slurms/passive_%j.out
#SBATCH --job-name=passive
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --exclusive
#SBATCH --time=02:00:00
#
# What breaks Elmer's passive elements in our model (the first shelf-cut runs diverged / gave NaN)? Short runs
# (2 steps, frozen grounding line, 32 ranks, 4 at a time) on the linear-bed test (configs/test_gl_bed.toml, its
# spun-up state), the Stokes elements beyond x = GL + DIST km (default 30) passive (ess run --shelf-cut):
#   Z  no cut, MUMPS (the reference: the full model works)
#   P  cut + Passive Target boundary with the shelf stress, GCR + ILU8 (the failing run again, Elmer output level 6)
#   Q  the same with MUMPS (a broken system, or only ILU breaking down?)
#   N  cut, NO Passive Target boundary (stress-free cut face), MUMPS (passive elements alone)
#   X  Q with the bed in x/y/z components (--cartesian-bed: no normal-tangential rotation)
#   M1 / M4  Elmer/Ice's own FS-SSA test (diag/mismip_fs_ssa, MISMIP 2D flowline) on 1 rank (UMFPACK) and 4 ranks
#      (MUMPS): do passive elements work in this Elmer build at all, serially and in parallel?
# Reading it: M1 fails -> Elmer's passive machinery is broken in this build (developers); M1 runs, M4 fails ->
# broken in parallel; M runs, N fails -> passive elements clash with something in our model; N runs, Q fails ->
# the Passive Target boundary; X runs, Q fails -> with the normal-tangential bed; Q runs, P fails -> ILU only.
#   sbatch --export=ALL,ELMER_MODULE=elmerfem/2026-10-28 jobs/passive_tests.cmd       (from ESS_solvers)
# ONLY_ELMER=1: only M1 and M4 (no ESS start state needed); ELMER_PREFIX: a self-built Elmer
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:?ELMER_MODULE not set}
export GL_CONFIG=${GL_CONFIG:-configs/test_gl_bed.toml}
WORK=work/test_gl_bed
START=${START:-$(ls $WORK/ref_iterative/spinup_chunks/points/state_*.npz 2>/dev/null | tail -1)}
[ -f "$START" ] || [ "$ONLY_ELMER" = 1 ] || { echo "no start state (START=...)"; exit 1; }
DIST=${DIST:-30}
L=slurms/passive_${SLURM_JOB_ID:-local}

bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
if [ -n "$ELMER_PREFIX" ]; then        # a self-built Elmer (jobs/build_elmer.cmd) in front of the module's
    export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
    export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$LD_LIBRARY_PATH
fi
echo "Elmer: $ELMER_MODULE -> $(command -v ElmerSolver)"
export ESS_NPROCS=32 ESS_MPIRUN="mpirun -np {n} --bind-to none"
if [ "$ONLY_ELMER" != 1 ]; then
    GL=$(python3 periodic_test/run_gl_ess.py glpos $START | tail -1)
    CUT=$(python3 -c "print($GL + $DIST)")
fi
echo "start $START, grounding line $GL km, passive beyond $CUT km"

# Elmer's own FS-SSA test, serial and on 4 ranks
D=$WORK/passive_mismip_${SLURM_JOB_ID:-local}
rm -rf $D; mkdir -p $D/m1 $D/m4
cp -r diag/mismip_fs_ssa/{coupled.sif,mesh2d,uppersurface.dat,ELMERSOLVER_STARTINFO} $D/m1/
cp -r diag/mismip_fs_ssa/{coupled.sif,mesh2d,uppersurface.dat,ELMERSOLVER_STARTINFO} $D/m4/
sed -i 's/umfpack/mumps/I' $D/m4/coupled.sif
( cd $D/m4 && ElmerGrid 2 2 mesh2d -metiskway 4 > partition.log 2>&1 )
( cd $D/m1 && timeout 3000 ElmerSolver coupled.sif > run.log 2>&1; echo "exit $?" >> run.log ) &
( cd $D/m4 && timeout 3000 mpirun -np 4 --bind-to none ElmerSolver coupled.sif > run.log 2>&1; echo "exit $?" >> run.log ) &

if [ "$ONLY_ELMER" = 1 ]; then          # only Elmer/Ice's own FS-SSA test
    wait || true
    for t in M1 M4; do
        d=$D/$(echo $t | tr M m)
        echo "$t (Elmer/Ice MISMIP_FS-SSA, $([ $t = M1 ] && echo '1 rank, UMFPACK' || echo '4 ranks, MUMPS')): $(tail -1 $d/run.log)"
        grep -i -m8 "passive\|reference norm\|passed\|failed\|error\|signal\|nan" $d/run.log | sed 's/^/      /'
    done
    exit 0
fi
opts="--state $START --stokes-only --chunks 1 --steps 2 --no-remask --bed cold --gl frozen --output-level 6"
run() {   # $1 letter, rest: ess run options
    local t=$1; shift
    python3 -u -m ess run $GL_CONFIG $opts --out $WORK/passive_$t.npz "$@" > ${L}_$t.log 2>&1
}
run Z --linear-stokes direct &
run P --linear-stokes iterative --stokes-preconditioner ILU8 --shelf-cut $CUT &
wait -n || true
run Q --linear-stokes direct --shelf-cut $CUT &
wait -n || true
run N --linear-stokes direct --shelf-cut $CUT --shelf-cut-no-target &
wait -n || true
run X --linear-stokes direct --shelf-cut $CUT --cartesian-bed &
wait || true

echo
echo "===== $(date +%T) ====="
for t in M1 M4; do
    d=$D/$(echo $t | tr M m)
    echo "$t (Elmer/Ice MISMIP_FS-SSA, $([ $t = M1 ] && echo '1 rank, UMFPACK' || echo '4 ranks, MUMPS')): $(tail -1 $d/run.log)"
    grep -i -m5 "reference norm\|passed\|failed\|error\|signal\|nan" $d/run.log | sed 's/^/      /'
done
for t in Z P Q N X; do
    case $t in Z) what="no cut, MUMPS";; P) what="cut + Passive Target, GCR + ILU8";; Q) what="cut + Passive Target, MUMPS";;
               N) what="cut, no Passive Target, MUMPS";; X) what="cut + Passive Target, MUMPS, x/y/z bed";; esac
    log=$(grep -o "see [^ ]*elmer.log" ${L}_$t.log | tail -1 | cut -c5-)
    if grep -q "wrote $WORK/passive_$t.npz" ${L}_$t.log; then
        echo "$t ($what): RAN   $(grep -o "Elmer [0-9]* s for .*" ${L}_$t.log | tail -1)"
        [ $t != Z ] && [ -f $WORK/passive_Z.npz ] && \
            python3 periodic_test/run_gl_ess.py cutcheck $WORK/passive_Z.npz $WORK/passive_$t.npz $CUT $GL | tail -1
    else
        echo "$t ($what): FAILED   $log"
        if [ -f "$log" ]; then
            echo "      passive boundary messages: $(grep -c "Passive element BC" $log)"
            # what Elmer was doing when it failed: the lines before the first error, without the backtrace
            n=$(grep -n -m1 "ERROR\|diverged\|Floating\|SIGSEGV\|signal" $log | cut -d: -f1)
            [ -n "$n" ] && head -$n $log | grep -v "^#\|^ *at \|^ *$\|libc\|???" | tail -25 | sed 's/^/      /'
        fi
    fi
done
