#!/bin/bash
# The coarse model's time-step series, mechanical problem only: Stokes, the free surfaces, the contact grounding line
# and the SSA shelf; NO heat, no strain heating, no cold/temperate mask, no water sheet (the temperature, and so the
# sliding pattern, frozen as in the start state). Which step sizes are stable, and how far from the dt5 reference,
# for full FSSA (one solve per step) and adapted FSSA (free surfaces in the coupled loop, 3 iterations), without the
# thermal coupling in the way.
#
#   bash jobs/mech_dt_series.sh                    (from the repo root, on the login node)
#   AFTER_A=31552 AFTER_B=31555 bash jobs/mech_dt_series.sh      (each lane waits for that job first)
#
# Two lanes, each a chain (a job starts when the previous one ends), so at most two run at once:
#   lane A: full dt5 (the reference)  -> adapted dt50 -> adapted dt20 -> adapted dt10
#   lane B: full dt50                  -> full dt20     -> full dt10
# YEARS (default 2500: the heat-coupled dt50 runs blew up by year ~950), the boundary VTU every 500 years
# (EVERY_YEARS), START_STATE (default the year-1505 checkpoint), NODES as the scaling point (default c640).
#
# A perturbation (the start state is near a mechanical steady state for its frozen temperature, a weak test): any
# job setting in the environment goes to every run, e.g.
#   FRIC=0.5 SUFFIX=_fric05 bash jobs/mech_dt_series.sh      (basal friction halved at t = 0: speed-up, thinning and
#                                                              grounding-line retreat over centuries)
# SUFFIX is added to every tag.
set -e
cd "$(dirname "$0")/.."
YEARS=${YEARS:-2500}; EVERY_YEARS=${EVERY_YEARS:-500}; P=${NODES:-c640}
B="START_STATE=${START_STATE:-work/coarse/ckpt_1500.npz} STEP_CHECK=1 ADAPTIVE_PREC=${ADAPTIVE_PREC:-ILU0:ILU1:ILU2} ADAPTIVE_PREC_CALM=${ADAPTIVE_PREC_CALM:-1000} RECOMPUTE=2 TL=${TL:-24:00:00}"

sub() {   # sub AFTER DT FSSA TAG -> job id
    local after=$1 dt=$2 fssa=$3 tag=$4 steps every extra
    steps=$(( YEARS / dt )); every=$(( EVERY_YEARS / dt )); [ $every -ge 1 ] || every=1
    extra=""
    [ "$fssa" = adapted ] && extra="FS_COUPLED=1 COUPLED_ITERS=3"
    tag=$tag${SUFFIX:-}
    env $B ${after:+AFTER=$after} FSSA=$fssa $extra DT=$dt STEPS=$steps VTU_EVERY=$every TAG=$tag \
        bash jobs/scaling_sweep.sh $P | awk -v t="$tag" '/^c[0-9]+:/{print $2; print "  " t ": " $2 > "/dev/stderr"}'
}

# ONLY="100:full 100:adapted ...": just these runs (dt:fssa), as ONE chain after AFTER (no lanes), e.g. extra step
# sizes for a series already run:  AFTER=31571 ONLY="100:full 100:adapted" SLIDING=temperate FRIC=100 SUFFIX=_unif100 ...
if [ -n "$ONLY" ]; then
    echo "mechanical-only runs: $ONLY, ${YEARS} yr each, $P${FRIC:+, friction x $FRIC}${SLIDING:+, sliding $SLIDING}"
    a=${AFTER:-}
    for r in $ONLY; do a=$(sub "$a" ${r%%:*} ${r##*:} _mech_dt${r%%:*}_${r##*:}); done
    echo "submitted (one chain)"; exit 0
fi
echo "mechanical-only time-step series: ${YEARS} yr each, VTU every ${EVERY_YEARS} yr, $P${FRIC:+, friction x $FRIC}${LUA:+, $LUA}"
a=${AFTER_A:-}
a=$(sub "$a" 5 full _mech_dt5_full)
a=$(sub "$a" 50 adapted _mech_dt50_adapted)
a=$(sub "$a" 20 adapted _mech_dt20_adapted)
a=$(sub "$a" 10 adapted _mech_dt10_adapted)
b=${AFTER_B:-}
b=$(sub "$b" 50 full _mech_dt50_full)
b=$(sub "$b" 20 full _mech_dt20_full)
b=$(sub "$b" 10 full _mech_dt10_full)
echo "submitted; squeue -u \$USER -o \"%.8i %.10T %E\" shows the chains"
