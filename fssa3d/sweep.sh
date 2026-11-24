#!/bin/bash
# Submit the free-surface comparison on the 3D strip (after `sbatch fssa3d/prep.cmd`): each variant starts from the
# extruded flowline steady state; the friction dip is applied from t = 0 (DIP > 0) and the response is compared across
# stabilisations (FSSA full / none / adapted) and the explicit vs implicit (coupled) free surface.
# DIP=0 runs are the controls: the extruded state should stay put (any drift is the 3D discretisation, not the dip).
#
#   bash fssa3d/sweep.sh                       # the default list below, one job each (64 ranks)
#   DT=50 YEARS=2000 DIP=0.5 bash fssa3d/sweep.sh
#   VARIANTS="full none" bash fssa3d/sweep.sh  # a subset
#
# Variant names: full | none | adapted (normal and transposed dropped) (explicit free surface); <name>+c = the same with the
# free surface in the coupled loop (FS_COUPLED=1, COUPLED_ITERS=3). Results: work/flowline_gl_3d/fssa_<TAG>.npz,
# python3 fssa3d/ess_compare.py.
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/flowline_gl_3d.toml}
WORK=$(python3 -c "import os; from ess.config import load_config as L; print(os.path.relpath(L('$CFG')['_work']))")
START=${START:-$WORK/start_flowline.npz}
[ -f "$START" ] || { echo "no $START: run fssa3d/prep.cmd first"; exit 1; }
DT=${DT:-10}; YEARS=${YEARS:-1000}; DIP=${DIP:-0.5}; NT=${NT:-64}; TIME=${TIME:-12:00:00}
VARIANTS=${VARIANTS:-"full none full+c adapted+c"}
for v in $VARIANTS $([ "$CONTROL" = 0 ] || echo control_full control_full+c); do
    d=$DIP; f=${v%+c}; c=0
    case $v in control_*) d=0; f=${f#control_};; esac
    [[ $v == *+c ]] && c=1
    TAG=dip${d}_dt${DT}_${f}$([ $c = 1 ] && echo _coupled3)
    [[ $v == control_* ]] && TAG=control_dt${DT}_${f}$([ $c = 1 ] && echo _coupled3)
    echo "submit $TAG"
    CFG=$CFG START=$START DT=$DT YEARS=$YEARS FSSA=$f FS_COUPLED=$c COUPLED_ITERS=3 TAG=$TAG \
    ESS_SLIDING=temperate ESS_SLIDE_YDIP=$d ESS_SLIDE_YDIP_WIDTH=${DIP_WIDTH:-2} \
        sbatch --ntasks=$NT --time=$TIME --job-name="f3d-$TAG" jobs/flowline_fssa.cmd
done
