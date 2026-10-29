#!/bin/bash
#SBATCH --output=./slurms/build_elmer_%j.out
#SBATCH --job-name=build_elmer
#SBATCH --ntasks=32
#SBATCH --time=03:00:00
#
# Build Elmer (with Elmer/Ice, MPI, OpenMP, Lua, MUMPS, Hypre) from source on tethys, against the system libraries
# the elmerfem modules use (Debian: Hypre 2.26, MUMPS 5.5, ScaLAPACK, OpenBLAS, OpenMPI). Default revision: the one
# of elmerfem/2026-10-28 (dd6aef5e2), so a build without patches behaves like the module.
#   1. on the LOGIN node (compute nodes may have no internet):   bash jobs/build_elmer.cmd fetch
#   2. build and install as a job:                               sbatch jobs/build_elmer.cmd
#   3. use it in ESS jobs: ELMER_PREFIX=/SCRATCH/$USER/elmer/install (with ELMER_MODULE for the MPI environment),
#      e.g. sbatch --export=ALL,ELMER_MODULE=elmerfem/2026-10-28,ELMER_PREFIX=/SCRATCH/$USER/elmer/install,... job
# ELMER_SRC / ELMER_BUILD / ELMER_PREFIX / ELMER_REV override the locations and the revision. Patches: every
# elmer/patches/*.patch of this repository is applied to the source before building (none yet: a vanilla build).
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
REPO=$PWD
mkdir -p slurms
ROOT=/SCRATCH/$USER/elmer
SRC=${ELMER_SRC:-$ROOT/src}
BUILD=${ELMER_BUILD:-$ROOT/build}
PREFIX=${ELMER_PREFIX:-$ROOT/install}
REV=${ELMER_REV:-dd6aef5e2}

if [ "$1" = fetch ]; then
    if [ ! -d $SRC/.git ]; then
        mkdir -p $(dirname $SRC)
        git clone https://github.com/ElmerCSC/elmerfem.git $SRC
    fi
    git -C $SRC fetch --all --tags || true
    git -C $SRC checkout -f $REV
    git -C $SRC submodule update --init --recursive || true
    echo "source $SRC at $(git -C $SRC log -1 --format='%h %cd')"
    exit 0
fi

[ -d $SRC/.git ] || { echo "no source in $SRC: run 'bash jobs/build_elmer.cmd fetch' on the login node first"; exit 1; }
module load ${ELMER_MODULE:-elmerfem/2026-10-28} 2>/dev/null || true   # the same MPI environment as the module runs
echo "building $SRC ($(git -C $SRC log -1 --format='%h %cd')) into $PREFIX on $(hostname), $(nproc) cores"
# a clean source at the revision, then the patches
git -C $SRC checkout -f $REV
git -C $SRC clean -fdq -e build
for p in $REPO/elmer/patches/*.patch; do
    [ -f "$p" ] || continue
    echo "applying $(basename $p)"
    git -C $SRC apply --whitespace=nowarn "$p"
done
rm -rf $BUILD; mkdir -p $BUILD
cd $BUILD
cmake $SRC \
    -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release \
    -DWITH_MPI=TRUE -DWITH_OpenMP=TRUE -DWITH_LUA=TRUE \
    -DWITH_Mumps=TRUE \
    -DWITH_Hypre=TRUE -DHypre_INCLUDE_DIR=/usr/include/hypre \
    -DHypre_LIBRARIES=/usr/lib/x86_64-linux-gnu/libHYPRE.so \
    -DWITH_ElmerIce=TRUE -DWITH_ELMERGUI=FALSE \
    > $REPO/slurms/build_elmer_${SLURM_JOB_ID:-local}_cmake.log 2>&1 \
    || { echo "cmake failed:"; tail -40 $REPO/slurms/build_elmer_${SLURM_JOB_ID:-local}_cmake.log; exit 1; }
grep -i -e "hypre" -e "mumps" -e "lua" -e "elmerice" $REPO/slurms/build_elmer_${SLURM_JOB_ID:-local}_cmake.log | head -20
echo "$(date +%T) make -j ${SLURM_NTASKS:-16}"
make -j ${SLURM_NTASKS:-16} install > $REPO/slurms/build_elmer_${SLURM_JOB_ID:-local}_make.log 2>&1 \
    || { echo "make failed:"; grep -n -m10 -i "error" $REPO/slurms/build_elmer_${SLURM_JOB_ID:-local}_make.log; tail -30 $REPO/slurms/build_elmer_${SLURM_JOB_ID:-local}_make.log; exit 1; }
echo "$(date +%T) installed in $PREFIX"
export PATH=$PREFIX/bin:$PATH LD_LIBRARY_PATH=$PREFIX/lib:$PREFIX/lib/elmersolver:$LD_LIBRARY_PATH ELMER_HOME=$PREFIX
echo "ElmerSolver: $(command -v ElmerSolver)"
ldd $(command -v ElmerSolver) | grep -i -e hypre -e mumps | awk '{print "   " $1 " -> " $3}'
# smoke test: Elmer's own BoomerAMG block test, serial
T=$BUILD/smoke_mgdyn; rm -rf $T; cp -r $REPO/diag/elmer_tests/mgdyn_airgap2 $T
( cd $T && ElmerSolver > run.log 2>&1 ); echo "smoke test mgdyn_airgap2 (serial): TEST.PASSED = $(cat $T/TEST.PASSED 2>/dev/null || echo missing)"
