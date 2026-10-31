#!/bin/bash
# Strong and weak scaling of the marine FS-SSA setup (10 km full-Stokes region + SSA shelf, contact grounding line,
# 2 Stokes solves per step, 1e-5) and strong scaling of the land benchmark, one jobs/scaling_tests.cmd job per point.
# Run on the login node: it only submits. Every point: (mesh setup and partitioning the first time), a ramped warm-up
# (R0, scaling_<ranks>_R0.npz in the config's work dir), then STEPS warm 1-yr steps whose summary gives the mean Stokes
# time per step. Points are chained (afterany: one at a time, a failed point does not stop the rest), so at most one
# point holds nodes; give fewer points at a time when the job limit is near.
#
#   bash jobs/scaling_sweep.sh [POINT ...]        (default: all)
# POINTs:  m96 m192 m384 m768   strong, marine, configs/test_periodic_gl_50km_fine.toml (192 x 4 boxes at 768, ...)
#          w192 w384            weak, marine: 12.5 km at 192, 25 km at 384 (the 50 km at 768 is m768)
#          l72 l144 l288 l576   strong, land, configs/test_wall_land_50km.toml (the marine's nodes per rank x 0.75)
# AFTER=<jobid>: the first point waits for that job. STEPS (default 10).
set -e
cd "$(dirname "$0")/.."
POINTS=${*:-m96 m192 m384 m768 w192 w384 l72 l144 l288 l576}
BASE="ALL,ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28},ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install2},SET=W,BED=none,TOL=1e-5,STEPS=${STEPS:-10},DT=1,BOXES=1,PY=4"
MAR="SLIDING=subtemperate,GLMODE=contact,NL=2,WARMUP=1e-2:0.1*3:1*3,SPRING_DT=1,FS_SSA=10,FS_SSA_SHELF=ssa,RECOMPUTE=20"
LAND="WARMUP=1*3,RECOMPUTE=10"
dep=${AFTER:+--dependency=afterany:$AFTER}
for p in $POINTS; do
    n=${p:1}
    case $p in
        m*) cfg=configs/test_periodic_gl_50km_fine.toml; extra=$MAR;;
        w192) cfg=configs/test_periodic_gl_12p5km_fine.toml; extra="$MAR,FROM=work/test_periodic_gl_50km/start.npz";;
        w384) cfg=configs/test_periodic_gl_25km_fine.toml; extra="$MAR,FROM=work/test_periodic_gl_50km/start.npz";;
        l*) cfg=configs/test_wall_land_50km.toml; extra=$LAND;;
        *) echo "unknown point $p"; exit 1;;
    esac
    w=$(python3 -c "import sys; sys.path.insert(0, '.'); from ess.config import load_config; print(load_config('$cfg')['_work'])" | tail -1)
    # FROM only when the mesh is not set up yet (scaling_tests.cmd sets it up and carries the state over)
    [ -f "$w/start.npz" ] && extra=$(echo "$extra" | sed 's/,FROM=[^,]*//')
    j=$(sbatch --parsable $dep --ntasks=$n --export=$BASE,GL_CONFIG=$cfg,RANKS=$n,$extra,TAG=_sweep jobs/scaling_tests.cmd)
    echo "$p: $j ($cfg, $n ranks)"
    dep="--dependency=afterany:$j"
done
echo "last job: $j   (summaries: grep 'mean Stokes' slurms/scaling_<jobid>.out)"
