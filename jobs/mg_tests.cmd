#!/bin/bash
#SBATCH --output=./slurms/mgtests_%j.out
#SBATCH --job-name=mgtests
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#
# Is the multigrid crash (corrupted heap / "double free" whenever BoomerAMG or Elmer's own cluster multigrid sets up
# in our Stokes runs) a problem of the INSTALLATION or of our model? Two of Elmer's own test cases
# (diag/elmer_tests: BlockLinElast3b = block solver with Elmer's cluster multigrid; mgdyn_airgap2 = block solver
# with Hypre BoomerAMG as BiCGStab's preconditioner; mgdyn_airgap2_amgsolver = the same with BoomerAMG as the
# block's solver, as ESS's crashing setup), each on 1 and on 4 ranks, with every Elmer module listed in MODULES. Elmer's developers run
# these tests for every revision, so on a sound build they pass.
# Reading it: they fail with the new module only -> that build (ask the installer); they fail with every module ->
# the installation environment (MPI, Hypre, compilers) or these tests on this system; they pass -> the crash needs
# something in our model (jobs/scaling_tests.cmd CHECKED=1 cleared ESS's own libraries).
#   sbatch jobs/mg_tests.cmd           (MODULES="elmerfem/2026-10-28 elmerfem/2026-07-13 elmerfem/2025-09-26")
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
MODULES=${MODULES:-elmerfem/2026-10-28 elmerfem/2026-07-13 elmerfem/2025-09-26}
D=work/mg_tests_${SLURM_JOB_ID:-local}
rm -rf $D; mkdir -p $D
declare -A RES
for m in $MODULES; do
    ( module unload elmerfem >/dev/null 2>&1; module load $m || exit 1
      echo "== $m -> $(command -v ElmerSolver)"
      echo "   linked: $(ldd $(command -v ElmerSolver) 2>/dev/null | grep -io 'lib[a-z_]*\(hypre\|mumps\|trilinos\|ml\)[a-z_]*\.so[.0-9]*' | sort -u | tr '\n' ' ')"
    )
    for t in BlockLinElast3b mgdyn_airgap2 mgdyn_airgap2_amgsolver; do
        for n in 1 4; do
            w=$D/$(echo $m | tr '/' '_')/${t}_np$n
            mkdir -p $w; cp -r diag/elmer_tests/$t/. $w/
            (
                module unload elmerfem >/dev/null 2>&1; module load $m || exit 1
                cd $w
                if [ $t = BlockLinElast3b ]; then
                    ElmerGrid 1 2 angle.grd > grid.log 2>&1
                    [ $n -gt 1 ] && ElmerGrid 2 2 angle -partdual -metiskway $n >> grid.log 2>&1
                else      # mgdyn_airgap2*
                    [ $n -gt 1 ] && ElmerGrid 2 2 airgap -partdual -metiskway $n -nooverwrite > grid.log 2>&1
                fi
                if [ $n = 1 ]; then timeout 900 ElmerSolver > run.log 2>&1
                else timeout 900 mpirun -np $n --bind-to none ElmerSolver > run.log 2>&1; fi
                echo "exit $?" >> run.log
            )
            pass=$(cat $w/TEST.PASSED $w/TEST.PASSED_$n 2>/dev/null | head -1)   # parallel runs write TEST.PASSED_<ranks>
            case "$pass" in 1) r=PASSED;; 0) r="FAILED (wrong result)";; *) r="CRASHED/ERROR";; esac
            RES["$m $t np$n"]="$r"
            echo "$(date +%T) $m $t np$n: $r"
        done
    done
done
echo
echo "===== summary ====="
for m in $MODULES; do
    for t in BlockLinElast3b mgdyn_airgap2 mgdyn_airgap2_amgsolver; do
        for n in 1 4; do
            w=$D/$(echo $m | tr '/' '_')/${t}_np$n
            echo "$m  $t  np$n:  ${RES["$m $t np$n"]}"
            case "${RES["$m $t np$n"]}" in PASSED) ;; *)
                grep -m4 -i "double free\|corrupt\|invalid pointer\|invalid size\|ERROR\|signal\|exit [1-9]" $w/run.log | sed 's/^/      /';; esac
        done
    done
done
