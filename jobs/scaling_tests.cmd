#!/bin/bash
#SBATCH --output=./slurms/scaling_%j.out
#SBATCH --job-name=scaling
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
#   G  F without Elmer's bandwidth optimisation (another elimination order: ILU0 pivots?)
#   H  F with row equilibration
#   K, L  ILU0 with Elmer's diagonal relaxation (Linear System ILU Factor 0.1, 1.0: the diagonal is multiplied by
#      1.1, 2 before factorising, the standard cure for ILU breakdown)
#   M, N  ILU0 with ILU Factor 0.01, 0.001     O  ILU1 with ILU Factor 0.01
#   P, Q, W, X  GCR + ILU2, ILU3, ILU4, ILU6 (the lowest fill that avoids the breakdown?)
#   Y, Z  Hypre FlexGMRES + Hypre ILU(0) coupling the partitions: RAS (overlapping, type 30), GMRES-ILU (Schur
#      complement on the interfaces, type 10)     V, I  the same with ILUT (types 31, 11; Hypre's default threshold)
#   A  GCR preconditioned by Elmer's own cluster (aggregation) multigrid on the whole system
#   B  the block preconditioner with the velocity block solved by Elmer's cluster multigrid
#   J  GCR with only the diagonal (Jacobi) as preconditioner: if it reduces the residual where ILU0 does not at
#      all, the ILU0 factors are broken somewhere (not merely weak)
# From the repository root, one job per partition count. No node limit: give --ntasks (default 128) and Slurm takes
# as many nodes as that needs; with --ntasks=RANKS the runs go one after another (more tasks than RANKS: several at
# once, but keep that within one node, as each mpirun starts on the first node). E.g. 512 ranks:
#   sbatch --ntasks=512 --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,RANKS=512 jobs/scaling_tests.cmd
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
# FROM=STATE: a config whose mesh does not exist yet (e.g. configs/test_wall_gl_50km.toml) is set up first and
# STATE carried onto it (ess setup, init, transfer -> <work dir>/start.npz)
if [ ! -f "$START" ] && [ -z "$FROM" ]; then
    echo "no start state in $WORK (START=..., or FROM=... to build the mesh): $(ls $WORK 2>&1 | head -20 | tr '\n' ' ')"; exit 1
fi
RANKS=${RANKS:-32}
SET=${SET:-RFSTUCDE}
DIST=${DIST:-30}
case $SET in *[CDE]*) case $SET in *R*) ;; *) SET="R$SET";; esac;; esac
L=slurms/scaling_${SLURM_JOB_ID:-local}

bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
if [ -n "$ELMER_PREFIX" ]; then        # a self-built Elmer (jobs/build_elmer.cmd) in front of the module's
    export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
    export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$LD_LIBRARY_PATH
fi
if [ ! -f "$START" ]; then
    echo "$(date +%T) setting up $GL_CONFIG for $RANKS ranks, start from $FROM (log ${L}_setup.log)"
    { ESS_NPROCS=$RANKS ESS_MPIRUN="mpirun -np {n} --bind-to none" python3 -u -m ess setup $GL_CONFIG && \
      ESS_NPROCS=$RANKS ESS_MPIRUN="mpirun -np {n} --bind-to none" python3 -u -m ess init $GL_CONFIG --out $WORK/state0.npz && \
      if [ "$FROM" = none ]; then cp $WORK/state0.npz $WORK/start.npz      # FROM=none: the config's own initial state
      else python3 -u -m ess transfer $GL_CONFIG --state $WORK/state0.npz --from $FROM --out $WORK/start.npz; fi; } \
        > ${L}_setup.log 2>&1 || { echo "setup failed:"; tail -30 ${L}_setup.log; exit 1; }
    START=$WORK/start.npz
fi
echo "Elmer: $ELMER_MODULE -> $(command -v ElmerSolver); $RANKS ranks per run; runs $SET; start $START"
echo "solver libraries linked to ElmerSolver: $(ldd $(command -v ElmerSolver) 2>/dev/null | grep -io 'lib[a-z_]*\(hypre\|trilinos\|epetra\|ml\|ifpack\|mumps\|petsc\)[a-z_]*' | sort -u | tr '\n' ' ')"
partition() {   # $1 case dir (with footprint.msh), $2 config: add footprint/partitioning.$RANKS, converted from
    # footprint.msh exactly as `ess setup` does (same node numbering; the mesh dir only holds partitioned copies).
    # The 2D footprint: Elmer extrudes it, so every rank gets whole columns.
    if [ "$RANKS" = 1 ]; then          # serial: the unpartitioned mesh files next to the partitionings
        [ -f $1/footprint/mesh.header ] && return 0
        local per1=""; grep -q "^periodic_sides *= *true" $2 && per1="-periodic 0 1 0"
        rm -rf $1/fp_serial
        ( cd $1 && ElmerGrid 14 2 footprint.msh -autoclean -out fp_serial $per1 ) > ${L}_partition.log 2>&1 \
            && cp $1/fp_serial/mesh.* $1/footprint/ && rm -rf $1/fp_serial && return 0
        cat ${L}_partition.log; return 1
    fi
    [ -d $1/footprint/partitioning.$RANKS ] && return 0
    echo "$(date +%T) partitioning $1/footprint for $RANKS ranks"
    local per="" part="-metiskway $RANKS"
    grep -q "^periodic_sides *= *true" $2 && per="-periodic 0 1 0"
    # box partitions RANKS/PY x PY for periodic sides, or with BOXES=1 (else METIS)
    if [ -n "$per" ] || [ "$BOXES" = 1 ]; then part="-partition $PX $PY 1"; fi
    rm -rf $1/fp_part$RANKS
    ( cd $1 && ElmerGrid 14 2 footprint.msh -autoclean -out fp_part$RANKS $per $part ) \
        > ${L}_partition.log 2>&1 || { cat ${L}_partition.log; return 1; }
    [ -d $1/fp_part$RANKS/partitioning.$RANKS ] || { echo "ElmerGrid made no partitioning.$RANKS:"; tail -20 ${L}_partition.log; return 1; }
    mv $1/fp_part$RANKS/partitioning.$RANKS $1/footprint/ && rm -rf $1/fp_part$RANKS
}
# box partitions for periodic sides: RANKS/PY x PY (PY default 4, as the production 64 x 4; PY=1: strips across the
# whole width, so every periodic node pair is in one partition). Partitionings are kept per rank count, so a
# different PY for a rank count already partitioned needs its partitioning.<ranks> directories removed first.
if [ -z "$PY" ]; then PY=1; [ $((RANKS % 4)) = 0 ] && PY=4; fi; PX=$((RANKS / PY))
YEL=$(grep -o "^y_elements *= *[0-9]*" $GL_CONFIG | grep -o "[0-9]*$" || echo 1)
( flock 9; partition $WORK/case $GL_CONFIG ) 9> $WORK/case/.lock
case $SET in *[CDE]*)        # the shelf cut needs a grounding line (marine)
    GL=$(python3 periodic_test/run_gl_ess.py glpos $START | tail -1)
    CUT=$(python3 -c "print($GL + $DIST)")
    echo "start grounding line $GL km; shelf cut (C, D, E) at $CUT km";;
esac
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
# CHECKED=1: ESS's own Elmer libraries built with bounds checks (make -C elmer checked) and glibc's heap checks on
# (MALLOC_CHECK_=3): memory corruption stops the run where it happens (file and line in the Elmer log)
if [ "$CHECKED" = 1 ]; then
    make -C elmer checked > ${L}_checked_build.log 2>&1 || { echo "checked build failed:"; tail -20 ${L}_checked_build.log; exit 1; }
    export ESS_ELMER_LIBDIR=$PWD/elmer/checked
    export ESS_MPIRUN="mpirun -np {n} --bind-to none -x MALLOC_CHECK_=3 -x MALLOC_PERTURB_=165"
    echo "CHECKED: ESS libraries from elmer/checked (bounds checks), glibc heap checks on"
fi
SLOTS=${SLOTS:-$(( ${SLURM_NTASKS:-128} / RANKS ))}; [ $SLOTS -ge 1 ] || SLOTS=1   # runs at a time (SLOTS=1: one by one)
# GLMODE=contact: the grounding line by contact every step (Elmer/Ice's SlidCoef_Contact: the sliding coefficient
# depends on the normal stress, so the Stokes problem is nonlinear; ElmerStreams' marine production, with NL=2);
# default frozen (not GL: that is the grounding-line position of the cut runs)
opts="--stokes-only --chunks 1 --steps ${STEPS:-2} --no-remask --gl ${GLMODE:-frozen} --linear-stokes block --output-level 4"
# BED=cold (default) | temperate (sliding everywhere) | none (the start state's own bed temperatures: the land
# benchmark's temperature-dependent sliding)
[ "${BED:-cold}" != none ] && opts="$opts --bed ${BED:-cold}"
# DT=years: the time step of every run (default the config's [map] dt, 5 yr for the 50 km tests); small steps as a
# production start (e.g. DT=1e-2) make the warm-started later solves representative
[ -n "$DT" ] && opts="$opts --dt $DT" && echo "time step $DT yr"
# DT_RAMP="1e-4:1e-2:0.1*3" (colons: sbatch --export splits at commas): a sequence of time steps (years, dt*n = n steps), as a production start; sets STEPS
if [ -n "$DT_RAMP" ]; then
    export ESS_DT_RAMP=$DT_RAMP
    STEPS=$(python3 -c "import re; print(sum(int(i.partition('*')[2] or 1) for i in re.split(r'[,:;\s]+', '$DT_RAMP') if i))")
    opts=$(echo "$opts" | sed "s/--steps [0-9]*/--steps $STEPS/")
    echo "time steps $DT_RAMP yr ($STEPS steps)"
fi
# SPRING_DT=years: the sea-pressure spring's time step fixed ([elmer] sea_spring_dt; default the run's dt, which makes
# the spring ~0 at tiny steps and Stokes nearly singular under the shelf)
[ -n "$SPRING_DT" ] && export ESS_SEA_SPRING_DT=$SPRING_DT && echo "sea spring time step fixed at $SPRING_DT yr"
# SLIDING=subtemperate|temperate: the sliding law for every run ([elmer] sliding; with BED=none the start state's
# bed temperatures decide where the bed slides)
[ -n "$SLIDING" ] && export ESS_SLIDING=$SLIDING && echo "sliding: $SLIDING"
# NL=n: Stokes nonlinear iterations per time step (ElmerStreams' marine production: 2, nonlinear tolerance 1e-5;
# default the template's 1)
[ -n "$NL" ] && export ESS_STOKES_NL_ITERS=$NL && echo "Stokes nonlinear iterations per step: up to $NL"
# KEEP_WARM=1: every run also saves its pressure, so its output can warm-start later runs (WARMUP_STATE=<its .npz>)
[ "$KEEP_WARM" = 1 ] && opts="$opts --add-diag Pressure"
# FS_SSA=km: the moving full-Stokes region ([elmer] fs_ssa_km): ice more than this beyond the current grounding line
# is passive for Stokes, its face carries the free-floating shelf's stress
[ -n "$FS_SSA" ] && export ESS_FS_SSA_KM=$FS_SSA && echo "moving full-Stokes region: up to $FS_SSA km beyond the grounding line"
# FS_SSA_TARGET=loads (default: the shelf force added to the Stokes right-hand side) | elmer (Elmer's Passive Target
# boundary) | 0 (none: the interface stress free; tests)
[ -n "$FS_SSA_TARGET" ] && export ESS_FS_SSA_TARGET=$FS_SSA_TARGET && echo "  interface shelf stress: $FS_SSA_TARGET"
# FS_SSA_SHELF=frozen (default) | ssa: the passive shelf by the shallow-shelf approximation every step ([elmer]
# fs_ssa_shelf: floated, moved with the SSA velocity, the interface stress from its strain rate)
[ -n "$FS_SSA_SHELF" ] && export ESS_FS_SSA_SHELF=$FS_SSA_SHELF && echo "  passive shelf: $FS_SSA_SHELF"
# TAG=name: the whole-shelf runs write <work>/scaling_<ranks><TAG>_<letter>.npz (keeps runs apart, e.g. TAG=_fsssa10)
# TOL: the GCR runs' convergence tolerance (default 1e-8)
[ -n "$TOL" ] && echo "GCR tolerance $TOL"
# STOKES_ELEMENT=Stabilized|Bubbles|P2/P1: the element for every run (default the config's). Low ILU needs
# Stabilized? (the flowline tests, where ILU0/ILU1 converged, were Stabilized; the 50 km config is Bubbles, where
# they did not reduce the residual at all)
[ -n "$STOKES_ELEMENT" ] && opts="$opts --stokes-element $STOKES_ELEMENT"
# ROT=1: the Stokes initial guess rotated to the normal-tangential frame before each solve (elmer/RotateGuess.F90;
# Elmer passes it unrotated, so warm starts begin far worse than zero at the bed nodes)
[ "$ROT" = 1 ] && opts="$opts --rotate-guess"
# CART=1: the bed condition in x/y/z components (marine template: ess run --cartesian-bed; no normal-tangential
# rotation of the bed unknowns)
[ "$CART" = 1 ] && opts="$opts --cartesian-bed"
# STOKES_SOLVER=flowsolve|ins: Elmer's older Stokes modules instead of IncompressibleNSVec
[ -n "$STOKES_SOLVER" ] && opts="$opts --stokes-solver $STOKES_SOLVER" && echo "Stokes module: $STOKES_SOLVER"
# MAXIT: the GCR iteration cap of the GCR runs (default 2000)
# FRIC=F: the sliding coefficient fcoeff times F (the periodic GL runs used 10, run_gl_ess.py --fbase; plain ess
# run, as here by default: 1)
if [ -n "$FRIC" ]; then
    FC=$(python3 -c "import sys; sys.path.insert(0, '.'); from ess.case import eval_lua_parameters as e; print(repr($FRIC * e('case/parameters.lua')['fcoeff']))" | tail -1)
    opts="$opts --lua fcoeff=$FC"; echo "sliding coefficient fcoeff x $FRIC = $FC"
fi
CFG_ELEMENT=$(grep -o '^stokes_element *= *"[^"]*"' $GL_CONFIG | cut -d'"' -f2)
echo "Stokes element: ${STOKES_ELEMENT:-$CFG_ELEMENT, from the config}"

V=$WORK/scaling_files_${SLURM_JOB_ID:-local}; mkdir -p $V
mkGCR() {   # $1 file, $2 ILU level: GCR + ILU on the whole Stokes system, every iteration printed (counts)
    printf '%s\n' "  Linear System Solver = Iterative" "  Linear System Iterative Method = GCR" \
        "  Linear System GCR Restart = Integer 250" "  Linear System Max Iterations = ${MAXIT:-2000}" \
        "  Linear System Preconditioning = ILU$2" "  Linear System Convergence Tolerance = ${TOL:-1.0e-8}" \
        "  Linear System Abort Not Converged = Logical False" "  Linear System Residual Output = 1" > $1
}
mkHILU() {   # $1 file, $2 Hypre ILU type: Hypre FlexGMRES + Hypre ILU on the whole Stokes system (fill level and drop
    # threshold at Hypre's defaults: Elmer's SolveHypre.c does not pass them)
    printf '%s\n' "  Linear System Solver = Iterative" "  Linear System Use Hypre = Logical True" \
        "  Linear System Iterative Method = FlexGMRES" "  Linear System Preconditioning = ILU$2" \
        "  Hypre GMRES Dimension = Integer 250" "  Linear System Max Iterations = ${MAXIT:-2000}" \
        "  Linear System Convergence Tolerance = 1.0e-8" "  Linear System Abort Not Converged = Logical False" > $1
}
MGKEYS=("MG Method = String cluster" "MG Levels = Integer 10" "MG Smoother = String sgs"
        "MG Pre Smoothing Iterations(1) = 1" "MG Post Smoothing Iterations(1) = 1" "MG Cluster Size = Integer 0"
        "MG Cluster Alpha = Real 1.8" "MG Strong Connection Limit = Real 0.01" "MG Max Iterations = Integer 1")
mkMG() {   # $1 file: GCR preconditioned by one cycle of Elmer's own cluster (aggregation) multigrid, whole system
    mkGCR $1 0
    sed -i 's/Linear System Preconditioning = ILU0/Linear System Preconditioning = Multigrid/' $1
    for k in "${MGKEYS[@]}"; do echo "  $k" >> $1; done
}
mkBlockMG() {   # $1 file: the block preconditioner, velocity block solved by Elmer's cluster multigrid (as Elmer's
    # BlockLinElast3b test), pressure block as before
    grep -v "block 11:" elmer/stokes_block_amg.sif.in > $1
    printf '%s\n' "  block 11: Linear System Solver = Multigrid" "  block 11: Linear System Convergence Tolerance = 1.0e-2" \
        "  block 11: Linear System Max Iterations = 20" "  block 11: Linear System Abort Not Converged = Logical False" >> $1
    for k in "${MGKEYS[@]}"; do echo "  block 11: $k" >> $1; done
    sed -i 's/block 11: MG Max Iterations = Integer 1/block 11: MG Max Iterations = Integer 20/' $1
}
# Narrowing the BoomerAMG-as-solver crash (lowercase runs): a = the MLB block setup as it is (elmer/stokes_block_amg.sif.in,
# the crashing one); c = a without Stokes-Vec's block preconditioner (no pressure Schur solver); b = a with scalar
# blocks (Block Structure 1 2 3 4: u, v, w each BoomerAMG, pressure as before, no block preconditioner);
# g = a without "BoomerAMG Smooth Type" (6: Schwarz smoothers); h = a with all BoomerAMG parameters at Hypre's
# defaults (only HYPRE Block Diagonal kept); d = a with "block 11: Linear System Preconditioning = None" given; e = a with
# the pressure block solved by MUMPS
mkAMGvar() {   # $1 file, $2 variant letter
    cp elmer/stokes_block_amg.sif.in $1
    case $2 in
        c) sed -i 's/Block Preconditioner = Logical True/Block Preconditioner = Logical False/' $1;;
        b) sed -i 's/Block Preconditioner = Logical True/Block Preconditioner = Logical False/; s/Block Structure(4) = Integer 1 1 1 2/Block Structure(4) = Integer 1 2 3 4/; s/block 22:/block 44:/' $1
           for k in 22 33; do grep "block 11:" elmer/stokes_block_amg.sif.in | grep -v "Num Functions\|Block Diagonal" | sed "s/block 11:/block $k:/" >> $1; done
           sed -i '/block 11: HYPRE Block Diagonal/d; /BoomerAMG Num Functions/d' $1;;
        g) sed -i '/BoomerAMG Smooth Type/d' $1;;
        h) sed -i '/^ *BoomerAMG /d; /^ *Boomeramg /d' $1;;
        d) echo "  block 11: Linear System Preconditioning = None" >> $1;;
        f) sed -i 's/Block Matrix Reuse = Logical False/Block Matrix Reuse = Logical True/' $1;;
        e) sed -i '/block 22:/d' $1
           printf '%s\n' "  block 22: Linear System Solver = Direct" "  block 22: Linear System Direct Method = MUMPS" >> $1;;
    esac
}
# With the patched Elmer (elmer/patches/01-hypre-options.patch, ELMER_PREFIX): n = block, velocity block Hypre
# BiCGStab + NODAL BoomerAMG (all three components of a node coarsened together; HMIS, extended interpolation,
# P max 4, threshold 0.5, inner 30 its / 1e-2); o = the same with BoomerAMG as the velocity SOLVER (the crash fix);
# p = n with nodal 1 and threshold 0.8; q, r = Hypre FlexGMRES + RAS-ILU(2) / GMRES-ILU(2) on the whole system
mkNodal() {   # $1 file, $2 nodal norm, $3 strong threshold, $4 "solver" for BoomerAMG as the solver
    grep -v "block 11:\|^ *BoomerAMG \|^ *Boomeramg " elmer/stokes_block_amg.sif.in > $1
    if [ "$4" = solver ]; then
        printf '%s\n' "  block 11: Linear System Solver = Iterative" "  block 11: Linear System Use Hypre = Logical True" \
            "  block 11: Linear System Iterative Method = BoomerAMG" >> $1
    else
        printf '%s\n' "  block 11: Linear System Solver = Iterative" "  block 11: Linear System Use Hypre = Logical True" \
            "  block 11: Linear System Iterative Method = BiCGStab" "  block 11: Linear System Preconditioning = BoomerAMG" >> $1
    fi
    printf '%s\n' "  block 11: Linear System Max Iterations = 30" "  block 11: Linear System Convergence Tolerance = 1.0e-2" \
        "  block 11: Linear System Abort Not Converged = Logical False" \
        "  BoomerAMG Num Functions = Integer 3" "  BoomerAMG Nodal = Integer $2" "  BoomerAMG Coarsen Type = Integer 10" \
        "  BoomerAMG Interpolation Type = Integer 6" "  BoomerAMG P Max Elmts = Integer 4" \
        "  BoomerAMG Relax Type = Integer 6" "  BoomerAMG Strong Threshold = Real $3" >> $1
}
mkHILUk() {   # $1 file, $2 Hypre ILU type, $3 fill level
    mkHILU $1 $2; echo "  ILU Fill Level = Integer $3" >> $1
}
# With the patched Elmer (elmer/patches/02-column-coarse.patch, ELMER_PREFIX): GCR with the two-level preconditioner
# "Linear System Column Coarse": a coarse level of column modes per mesh column (default order 1: u, v, w each
# uniform and linear in the relative height, plus a uniform pressure: plug flow that satisfies incompressibility,
# and vertical shear), solved by MUMPS on rank 0, then the usual partition-local ILU/diagonal on the remaining
# residual. j = l with pre-smoothing (ILU, coarse, ILU: a V(1,1) cycle); CC_ORDER=0: the first version's uniform
# u, v (+p) only. x = ILU0 + coarse; y = x without the pressure mode; z = Diagonal + coarse; l = ILU4 + coarse;
# m = ILU0 + ILU Factor 0.01 + coarse; i = the coarse level alone (no local preconditioner: tests the coarse
# step by itself). The Elmer log reports for the first 3 applications per solve what fraction of the residual the
# coarse step leaves (|r - A x1|/|r|) and the sizes of the coarse and local corrections. CC_RANKS: MUMPS ranks for the coarse problem (default 1)
mkCC() {   # $1 file, $2 preconditioner (ILU0, Diagonal, ...), $3 pressure mode True/False
    mkGCR $1 0
    sed -i "s/Linear System Preconditioning = ILU0/Linear System Preconditioning = $2/" $1
    printf '%s\n' "  Linear System Column Coarse = Logical True" "  Linear System Column Coarse Pressure = Logical $3" \
        "  Linear System Column Coarse Ranks = Integer ${CC_RANKS:-1}" \
        "  Linear System Column Coarse Order = Integer ${CC_ORDER:-1}" >> $1
    # CC_RECOMPUTE=n: the coarse operator and its factors rebuilt every n-th solve (default 1)
    [ -n "$CC_RECOMPUTE" ] && echo "  Linear System Column Coarse Recompute = Integer $CC_RECOMPUTE" >> $1
    # CC_SELECT=1: coarse modes only in floating/grounding-line columns (GroundedMask < 0.5) and over a temperate bed
    # (ColdTempMask > 0), grown by CC_GROW (default 2) rings of neighbouring columns
    if [ "$CC_SELECT" = 1 ]; then
        printf '%s\n' "  Linear System Column Coarse Below Variable = String GroundedMask" \
            "  Linear System Column Coarse Below Value = Real 0.5" \
            "  Linear System Column Coarse Above Variable = String ColdTempMask" \
            "  Linear System Column Coarse Above Value = Real 0.0" \
            "  Linear System Column Coarse Grow = Integer ${CC_GROW:-2}" >> $1
    fi
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
        G) mkGCR $V/G.sif.in 0; echo "  Optimize Bandwidth = Logical False" >> $V/G.sif.in;;
        H) mkGCR $V/H.sif.in 0; echo "  Linear System Row Equilibration = Logical True" >> $V/H.sif.in;;
        K) mkGCR $V/K.sif.in 0; echo "  Linear System ILU Factor = Real 0.1" >> $V/K.sif.in;;
        L) mkGCR $V/L.sif.in 0; echo "  Linear System ILU Factor = Real 1.0" >> $V/L.sif.in;;
        M) mkGCR $V/M.sif.in 0; echo "  Linear System ILU Factor = Real 0.01" >> $V/M.sif.in;;
        N) mkGCR $V/N.sif.in 0; echo "  Linear System ILU Factor = Real 0.001" >> $V/N.sif.in;;
        O) mkGCR $V/O.sif.in 1; echo "  Linear System ILU Factor = Real 0.01" >> $V/O.sif.in;;
        a|b|c|d|e|f|g|h) mkAMGvar $V/$t.sif.in $t;;
        k) grep -v "^#" elmer/stokes_block.sif.in > $V/k.sif.in;;
        n) mkNodal $V/n.sif.in 4 0.5;;  o) mkNodal $V/o.sif.in 4 0.5 solver;;  p) mkNodal $V/p.sif.in 1 0.8;;
        q) mkHILUk $V/q.sif.in 30 2;;  r) mkHILUk $V/r.sif.in 10 2;;
        s|t|u|v|w) mkNodal $V/$t.sif.in 1 0.8
           case $t in
             s) sed -i 's/BoomerAMG Strong Threshold = Real 0.8/BoomerAMG Strong Threshold = Real 0.9/' $V/s.sif.in;;
             t) sed -i 's/BoomerAMG Relax Type = Integer 6/BoomerAMG Relax Type = Integer 8/' $V/t.sif.in
                echo "  BoomerAMG Num Sweeps = Integer 2" >> $V/t.sif.in;;
             u) sed -i 's/block 11: Linear System Max Iterations = 30/block 11: Linear System Max Iterations = 100/; s/block 11: Linear System Convergence Tolerance = 1.0e-2/block 11: Linear System Convergence Tolerance = 1.0e-4/' $V/u.sif.in;;
             v) sed -i 's/block 11: Linear System Max Iterations = 30/block 11: Linear System Max Iterations = 300/; s/block 11: Linear System Convergence Tolerance = 1.0e-2/block 11: Linear System Convergence Tolerance = 1.0e-6/' $V/v.sif.in;;
             w) sed -i 's/BoomerAMG Strong Threshold = Real 0.8/BoomerAMG Strong Threshold = Real 0.9/; s/block 11: Linear System Max Iterations = 30/block 11: Linear System Max Iterations = 100/; s/block 11: Linear System Convergence Tolerance = 1.0e-2/block 11: Linear System Convergence Tolerance = 1.0e-4/' $V/w.sif.in;;
           esac;;
        A) mkMG $V/A.sif.in;;  B) mkBlockMG $V/B.sif.in;;
        x) mkCC $V/x.sif.in ILU0 True;;  y) mkCC $V/y.sif.in ILU0 False;;  z) mkCC $V/z.sif.in Diagonal True;;
        l) mkCC $V/l.sif.in ILU4 True;;  i) mkCC $V/i.sif.in None True;;
        j) mkCC $V/j.sif.in ILU4 True; echo "  Linear System Column Coarse Pre Smoothing = Logical True" >> $V/j.sif.in;;
        m) mkCC $V/m.sif.in ILU0 True; echo "  Linear System ILU Factor = Real 0.01" >> $V/m.sif.in;;
        Y) mkHILU $V/Y.sif.in 30;;  Z) mkHILU $V/Z.sif.in 10;;  V) mkHILU $V/V.sif.in 31;;  I) mkHILU $V/I.sif.in 11;;
        P) mkGCR $V/P.sif.in 2;;  Q) mkGCR $V/Q.sif.in 3;;  W) mkGCR $V/W.sif.in 4;;  X) mkGCR $V/X.sif.in 6;;
        J) mkGCR $V/J.sif.in 0; sed -i 's/Linear System Preconditioning = ILU0/Linear System Preconditioning = Diagonal/' $V/J.sif.in;;
        C) mkGCR $V/C.sif.in 8;;  D) mkGCR $V/D.sif.in 0;;  E) mkGCR $V/E.sif.in 1;;
        S) mkBlock $V/S.sif.in ParaSails;;
        T) mkBlock $V/T.sif.in FSAI;;
        U) mkBlock $V/U.sif.in BoomerAMG
           printf '%s\n' "  block 11: HYPRE Block Diagonal = Logical True" "  block 11: BoomerAMG Coarsen Type = Integer 10" \
               "  block 11: BoomerAMG Interpolation Type = Integer 6" "  block 11: BoomerAMG Strong Threshold = Real 0.5" >> $V/U.sif.in;;
        *) echo "unknown run $t"; exit 1;;
    esac
    # Elmer times the Stokes solver in every time step (assembly + linear solves) and each linear solve, so the
    # summary can give the cost per step without Elmer's start-up (needs output level 4)
    printf '%s\n' "  Linear System Timing = Logical True" "  Solver Timing = Logical True" >> $V/$t.sif.in
    # RECOMPUTE=n: the ILU factors rebuilt only every n-th linear solve (Elmer's Linear System Precondition
    # Recompute), reused in between: the factorisation is ~6 s of a ~10 s land step
    [ -n "$RECOMPUTE" ] && echo "  Linear System Precondition Recompute = Integer $RECOMPUTE" >> $V/$t.sif.in
    echo "! ESS scaling $t ${SLURM_JOB_ID:-local}" >> $V/$t.sif.in
done

export ESS_KEEP_RUNS=1 ESS_NO_CACHE=1     # every run really runs, and its Elmer log stays (iteration counts)
# WARM=1: a cold GCR + ILU8 run (R0) first, then every whole-shelf run starts its Stokes solves from R0's final
# velocity and pressure (ess run --warm-start), like a later time step of a long run (the cut runs C, D, E: cold,
# their mesh differs)
WARMOPT=""
if [ "$WARM" = 1 ]; then
    mkGCR $V/R0.sif.in 8; printf '%s\n' "  Linear System Timing = Logical True" "  Solver Timing = Logical True" "! ESS scaling R0 ${SLURM_JOB_ID:-local}" >> $V/R0.sif.in
    echo "$(date +%T) start R0 (cold, the warm start for the others)"
    python3 -u -m ess run $GL_CONFIG --state $RSTART $opts --add-diag Pressure --out $WORK/scaling_${RANKS}_R0.npz \
        --stokes-block-file $PWD/$V/R0.sif.in > ${L}_R0.log 2>&1 || { echo "R0 failed: ${L}_R0.log"; exit 1; }
    WARMOPT="--warm-start $WORK/scaling_${RANKS}_R0.npz"
fi
# WARMUP="1e-4:1e-2:0.1*3:1*2" (time steps, years): a production-like continuation. First a ramp-up run R0 (GCR +
# ILU4, the rotated guess, these steps; it also saves the final velocity and pressure), then every run CONTINUES
# from R0's end: its geometry and temperature (--state) and its velocity and pressure as the Stokes initial guess
# (--warm-start, with the rotated guess), with STEPS (default 5) steps of DT (default 1) years. The summary's
# per-step Stokes times are then those of a long run's ordinary steps. WARMUP_STATE=FILE: reuse an R0 state
# (a WARMUP job's work/<config>/scaling_<ranks>_R0.npz) instead of running the ramp again.
if [ -n "$WARMUP" ] || [ -n "$WARMUP_STATE" ]; then
    R0S=${WARMUP_STATE:-$WORK/scaling_${RANKS}_R0.npz}
    if [ -z "$WARMUP_STATE" ]; then
        mkGCR $V/R0.sif.in 4
        printf '%s\n' "  Linear System Timing = Logical True" "  Solver Timing = Logical True" "! ESS scaling R0 ${SLURM_JOB_ID:-local}" >> $V/R0.sif.in
        n0=$(python3 -c "import re; print(sum(int(i.partition('*')[2] or 1) for i in re.split(r'[,:;\s]+', '$WARMUP') if i))")
        o0=$(echo "$opts" | sed "s/--steps [0-9]*/--steps $n0/")
        echo "$(date +%T) start R0: the ramp-up $WARMUP yr ($n0 steps), GCR + ILU4 (log ${L}_R0.log)"
        ESS_DT_RAMP=$WARMUP python3 -u -m ess run $GL_CONFIG --state $RSTART $o0 --rotate-guess --add-diag Pressure \
            --out $R0S --stokes-block-file $PWD/$V/R0.sif.in > ${L}_R0.log 2>&1 || { echo "R0 failed: ${L}_R0.log"; exit 1; }
    fi
    [ -f "$R0S" ] || { echo "no warm-up state $R0S"; exit 1; }
    unset ESS_DT_RAMP
    opts=$(echo "$opts" | sed "s/--steps [0-9]*/--steps ${STEPS:-5}/; s/--dt [^ ]*//") 
    opts="$opts --dt ${DT:-1} --rotate-guess"
    RSTART=$R0S
    WARMOPT="--warm-start $R0S"
    echo "runs continue from $R0S: ${STEPS:-5} steps of ${DT:-1} yr, warm-started (velocity, pressure; rotated guess)"
fi
running=0
for t in $(echo $SET | fold -w1); do
    c="run $GL_CONFIG --state $RSTART $opts $WARMOPT"; o=$WORK/scaling_${RANKS}${TAG}_$t.npz
    case $t in C|D|E) c="run $CUTCFG --state $CW/start.npz $opts"; o=$CW/scaling_${RANKS}_$t.npz;; esac
    case $t in Y|Z|V|I|q|r) c="$c --output-level 6";; esac          # Hypre reports its iterations from level 6
    if [ $running -ge $SLOTS ]; then wait -n || true; running=$((running - 1)); fi
    echo "$(date +%T) start $t"
    python3 -u -m ess $c --out $o --stokes-block-file $PWD/$V/$t.sif.in > ${L}_$t.log 2>&1 &
    running=$((running + 1))
done
wait || true

echo
echo "===== $RANKS ranks, $(date +%T)$([ "$WARM" = 1 ] && echo ', WARM START from R0 (whole-shelf runs)')$([ -n "$WARMUP$WARMUP_STATE" ] && echo ", runs CONTINUE from the ramp-up R0 (${STEPS:-5} x ${DT:-1} yr)") ====="
for t in $([ "$WARM" = 1 ] || [ -n "$WARMUP" ] && echo R0) $(echo $SET | fold -w1); do
    case $t in R0) what="the warm-up run ($([ -n "$WARMUP" ] && echo "ramp $WARMUP, ILU4" || echo "GCR + ILU8, cold"))";; R) what="GCR + ILU8, whole shelf (reference)";; F) what="GCR + ILU0, whole shelf";;
               G) what="GCR + ILU0, whole shelf, no bandwidth optimisation (another elimination order)";;
               H) what="GCR + ILU0, whole shelf, row equilibration";;
               K) what="GCR + ILU0, diagonal x 1.1 before factorising (ILU Factor 0.1)";;
               L) what="GCR + ILU0, diagonal x 2 before factorising (ILU Factor 1.0)";;
               M) what="GCR + ILU0, ILU Factor 0.01";; N) what="GCR + ILU0, ILU Factor 0.001";;
               O) what="GCR + ILU1, ILU Factor 0.01";;
               k) what="block preconditioner, ILU inner solves (elmer/stokes_block.sif.in)";;
               n) what="block, velocity BiCGStab + NODAL BoomerAMG (patched Elmer)";;
               o) what="block, NODAL BoomerAMG as the velocity solver (patched Elmer: the crash fix)";;
               p) what="n with nodal 1, strong threshold 0.8";;
               q) what="Hypre FlexGMRES + RAS-ILU(2) (patched Elmer: fill level)";;
               r) what="Hypre FlexGMRES + GMRES-ILU(2) (patched Elmer: fill level)";;
               s) what="p with strong threshold 0.9";;
               t) what="p with l1-symmetric Gauss-Seidel (relax 8), 2 sweeps";;
               u) what="p with the velocity solves to 1e-4 (100 its)";;
               v) what="p with the velocity solves to 1e-6 (300 its): is the outer stall from the inexact inner solve?";;
               w) what="p with threshold 0.9 and velocity solves to 1e-4";;
               a) what="MLB block + BoomerAMG as the velocity block's solver, as is (crashed)";;
               b) what="a with scalar velocity blocks (1 2 3 4), no block preconditioner";;
               c) what="a without the block preconditioner (no pressure Schur solver)";;
               d) what="a with block 11 preconditioning explicitly None";;
               e) what="a with the pressure block (22) solved directly by MUMPS";;
               f) what="a with Block Matrix Reuse = True (as Elmer's passing BoomerAMG test)";;
               g) what="a without BoomerAMG Smooth Type (Schwarz smoothers)";;
               h) what="a with all BoomerAMG parameters at Hypre's defaults";;
               x) what="GCR + column coarse level + ILU0 (patched Elmer)";;
               y) what="x without the coarse pressure mode (velocity only)";;
               z) what="GCR + column coarse level + Diagonal";;
               l) what="GCR + column coarse level + ILU4";;
               i) what="GCR + column coarse level alone (no local preconditioner)";;
               j) what="l with pre-smoothing (ILU4, coarse, ILU4)";;
               m) what="x with ILU Factor 0.01";;
               A) what="GCR + Elmer's cluster multigrid (one cycle, SGS smoothing) on the whole system";;
               B) what="block, velocity block by Elmer's cluster multigrid";;
               Y) what="Hypre FlexGMRES + RAS-ILU(0) (overlapping partitions)";;
               Z) what="Hypre FlexGMRES + GMRES-ILU(0) (partitions coupled through a Schur complement)";;
               V) what="Hypre FlexGMRES + RAS-ILUT";; I) what="Hypre FlexGMRES + GMRES-ILUT";;
               P) what="GCR + ILU2";; Q) what="GCR + ILU3";; W) what="GCR + ILU4";; X) what="GCR + ILU6";;
               J) what="GCR + Jacobi (diagonal), whole shelf: progress where ILU0 makes none = ILU0 broken";;
               S) what="block, velocity block BiCGStab + ParaSails";; T) what="block, velocity block BiCGStab + FSAI";;
               U) what="block, velocity block BiCGStab + BoomerAMG on uncoupled components";;
               C) what="GCR + ILU8, mesh ends at $CUT km";; D) what="GCR + ILU0, shelf cut";;
               E) what="GCR + ILU1, shelf cut";; esac
    W=$WORK; case $t in C|D|E) W=$CW;; esac
    OUTN=scaling_${RANKS}${TAG}_$t.npz; case $t in C|D|E) OUTN=scaling_${RANKS}_$t.npz;; esac
    if grep -q "wrote $W/$OUTN" ${L}_$t.log; then
        echo "$t ($what): RAN   $(grep -o "Elmer [0-9]* s for .*" ${L}_$t.log | tail -1)"
        f=$(grep -l "ESS scaling $t ${SLURM_JOB_ID:-local}\$" $W/runs/*/run.sif 2>/dev/null | tail -1)
        if [ -n "$f" ]; then
            el=$(dirname $f)/elmer.log
            # outer iterations per Stokes solve (the counter restarts at every solve) and the final residual
            grep "gcr:" $el | awk '$2 < prev {n++; printf "      solve %d: %4d iterations, residual %s\n", n, prev, res}
                                  {prev=$2; res=$3} END {n++; printf "      solve %d: %4d iterations, residual %s\n", n, prev, res}'
            hc=$(grep -c "did not converge" $el || true); [ "$hc" -gt 0 ] && echo "      Hypre 'did not converge' messages: $hc"
            grep "SolveHypre: Required iterations" $el | head -4 | sed 's/^/      /'
            # Stokes cost per time step (Elmer's own timers, start-up excluded): the whole solver (assembly, nonlinear
            # iterations, linear solves) and the linear solves alone, wall-clock seconds
            st=$(grep "Solver time (CPU,REAL)" $el | awk '{print $NF == "(s)" ? $(NF-1) : $NF}' | tr '\n' ' ')
            ls=$(grep "Linear system time (CPU,REAL)" $el | awk '{print $(NF-1)}' | tr '\n' ' ')
            [ -n "$st" ] && echo "      Stokes solver per step (s): $st"
            [ -n "$ls" ] && echo "      linear solves (s):           $ls"
            [ -n "$st" ] && echo "$st" | awk '{if (NF > 1) {s = 0; for (i = 2; i <= NF; i++) s += $i; printf "      mean Stokes time of steps 2-%d (warm): %.1f s\n", NF, s/(NF-1)}}'
            grep "ColumnCoarse:" $el | grep -v "apply [0-9]" | head -3 | sed 's/^/      /'
        fi
        case $t in C|D|E) [ -f $WORK/scaling_${RANKS}_R.npz ] && \
            python3 periodic_test/run_gl_ess.py cutcheck $WORK/scaling_${RANKS}_R.npz $CW/scaling_${RANKS}_$t.npz $CUT $GL | tail -1;; esac
    else
        log=$(grep -o "see [^ ]*elmer.log" ${L}_$t.log | tail -1 | cut -c5-)
        echo "$t ($what): FAILED   $log"
        [ -n "$log" ] && grep -m3 -i "double free\|corruption\|error\|signal\|fatal" $log | sed 's/^/      /'
    fi
done
