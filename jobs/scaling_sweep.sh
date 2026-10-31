#!/bin/bash
# Strong and weak scaling of the marine FS-SSA setup and strong scaling of the land benchmark, on the production-style
# footprint (unstructured triangles mirrored about y = W/2, METIS partitions: ElmerGrid -metisper), one
# jobs/scaling_tests.cmd job per point. Run on the login node: it only submits.
#
# Marine (the settled setup, 2026-10-31): 10 km full-Stokes region renewed per 1 km of grounding-line motion, SSA
# shelf (iterative), contact grounding line, 2 Stokes solves per step, 1e-5, GCR + ILU0 (SET=F) with ILU0 rebuilds on
# the ranks whose grounded mask changed and no scheduled refresh (RECOMPUTE=200), the bedrock/filler mesh update only at
# the first step (the default for both). Land: GCR + ILU0 and GCR + ILU4 (SET=FW), the same refresh setting, iterative free surface.
# Every point: (mesh setup and partitioning the first time), a ramped warm-up with ILU0 (R0, scaling_<ranks>_R0.npz in the
# config's work dir; ILU4 overflowed on the 96-rank partitions), then STEPS warm 1-yr steps whose summary gives the mean Stokes time, the whole Elmer time per step
# ("Elmer ... s per step") and every solver's time per step (TIMING=1). Points are chained (afterany: one at a time, a
# failed point does not stop the rest), so at most one point holds nodes (<= 6 nodes at 768 ranks).
#
#   bash jobs/scaling_sweep.sh [POINT ...]        (default: all)
# POINTs:  m96 m192 m384 m768   strong, marine, configs/test_periodic_gl_50km_tri.toml
#          w192 w384            weak, marine: 12.5 km at 192, 25 km at 384 (the 50 km at 768 is m768)
#          l72 l144 l288 l576   strong, land, configs/test_wall_land_50km_tri.toml (~0.7x the marine domain: the
#                               marine's nodes per rank at 0.75x the ranks)
# AFTER=<jobid>: the first point waits for that job. STEPS (default 20).
set -e
cd "$(dirname "$0")/.."
POINTS=${*:-m96 m192 m384 m768 w192 w384 l72 l144 l288 l576}
BASE="ALL,ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28},ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install2},BED=none,TOL=1e-5,STEPS=${STEPS:-20},DT=1,TIMING=1,RECOMPUTE=200,WARMUP_ILU=0"
# MAR_SET (default F): the marine runs per point, e.g. FPQ = GCR + ILU0, ILU2, ILU3 (land showed ILU0 breaking down
# below ~300 columns per rank: 5 -> 60-89 iterations from 288 to 576 ranks, ILU4 stayed at 2-3). MAR_REBUILD_PREC (default
# ILU0): the level of the forced rebuilds on ranks whose grounded mask changed; "same" = each run's own level
MRP=${MAR_REBUILD_PREC:-ILU0}; [ "$MRP" = same ] && MRP= || MRP=",REBUILD_PREC=$MRP"
MAR="SET=${MAR_SET:-F},SLIDING=subtemperate,GLMODE=contact,NL=2,WARMUP=1e-2:0.1*3:1*3,SPRING_DT=1,FS_SSA=10,FS_SSA_SHELF=ssa,FS_SSA_UPDATE=1$MRP"
# LAND_SET (default FW: GCR + ILU0 and GCR + ILU4 per point). ILU0 wins at 288 ranks (1.9 vs 2.5 s Stokes) but its
# iterations grow and it gets slower at 576 (3.2 s); ILU4 keeps 2-3 iterations (1.5 s at 576) but overflows Elmer's
# integers on the large partitions (72, 144 ranks)
LAND="SET=${LAND_SET:-FW},WARMUP=1*3"
dep=${AFTER:+--dependency=afterany:$AFTER}
for p in $POINTS; do
    n=${p:1}
    case $p in
        m*) cfg=configs/test_periodic_gl_50km_tri.toml; extra="$MAR,FROM=work/test_periodic_gl_50km/start.npz";;
        w192) cfg=configs/test_periodic_gl_12p5km_tri.toml; extra="$MAR,FROM=work/test_periodic_gl_50km/start.npz";;
        w384) cfg=configs/test_periodic_gl_25km_tri.toml; extra="$MAR,FROM=work/test_periodic_gl_50km/start.npz";;
        l*) cfg=configs/test_wall_land_50km_tri.toml; extra="$LAND,FROM=work/test_wall_land_50km/start.npz";;
        *) echo "unknown point $p"; exit 1;;
    esac
    w=$(python3 -c "import sys; sys.path.insert(0, '.'); from ess.config import load_config; print(load_config('$cfg')['_work'])" | tail -1)
    # FROM only when the mesh is not set up yet (scaling_tests.cmd sets it up and carries the state over); each rank
    # count gets its own partitioning (partitioning.<ranks>) and warm-up (scaling_<ranks>_R0.npz)
    [ -f "$w/start.npz" ] && extra=$(echo "$extra" | sed 's/,FROM=[^,]*//')
    # a warm-up already done at this rank count is reused (WARMUP_STATE) instead of ramping again
    [ -f "$w/scaling_${n}_R0.npz" ] && extra=$(echo "$extra" | sed "s/,WARMUP=[^,]*//"),WARMUP_STATE=$w/scaling_${n}_R0.npz
    # wall-time limit: fewer ranks take longer (a 768-rank point ~25 min incl. warm-up; 96 ranks ~8x the work per rank)
    tl=03:00:00; [ "$n" -le 192 ] && tl=06:00:00; [ "$n" -le 96 ] && tl=10:00:00
    j=$(sbatch --parsable $dep --ntasks=$n --time=$tl --export=$BASE,GL_CONFIG=$cfg,RANKS=$n,$extra,TAG=_sweep jobs/scaling_tests.cmd)
    echo "$p: $j ($cfg, $n ranks)"
    dep="--dependency=afterany:$j"
done
echo "last job: $j   (summaries: grep 'Elmer .* per step\|mean Stokes' slurms/scaling_<jobid>.out)"
