#!/bin/bash
# 3D pages of the free-surface comparison: each dip run beside its control (one copy, no temperature, no tiling):
#   full    vs control_full
#   none    vs control_full      (there is no control without stabilisation: control_full is the no-dip reference)
#   adapted vs control_adapted
# Packs the VTU outputs of runs still going first (re-run to add new steps), then writes the pages into $W:
#   full_vs_control.html  none_vs_control.html  adapted_vs_control.html
# Defaults: the jobs of the sweep (J_ADAPTED, J_CONTROL_FULL, J_CONTROL_ADAPTED, J_FULL, J_NONE; X0 X1 km window).
#   bash fssa3d/pages.sh                      (then the rsync line it prints, on the Mac)
cd "$(dirname "$0")/.." || exit 1
W=${W:-work/flowline_gl_3d_t}
J_FULL=${J_FULL:-32332}; J_NONE=${J_NONE:-32333}; J_ADAPTED=${J_ADAPTED:-32334}
J_CF=${J_CONTROL_FULL:-32335}; J_CA=${J_CONTROL_ADAPTED:-32336}
X0=${X0:-700}; X1=${X1:-840}

# frames of a job: the finished run's own file (fssa_<TAG>_frames.npz) when it exists, else packed from the run directory
frames() {   # JOB TAGPATTERN -> prints the file
    local J=$1 PAT=$2 F
    F=$(ls $W/fssa_${PAT}_frames.npz 2>/dev/null | head -1)
    if [ -z "$F" ] || ! [ -s "$F" ] || squeue -u "$USER" -h -o %i 2>/dev/null | grep -qx "$J"; then
        RUN=$(ls -d $W/runs/*fssa${J}*/ 2>/dev/null | head -1)
        if [ -n "$RUN" ]; then F=$W/fssa_job${J}_frames.npz; python3 plots/vtu_frames.py "$RUN" "$F" >&2; fi
    fi
    echo "$F"
}
FULL=$(frames $J_FULL "dip0.5_dt10_full")
NONE=$(frames $J_NONE "dip0.5_dt10_none")
ADA=$(frames $J_ADAPTED "dip0.5_dt10_adapted_coupled3")
CF=$(frames $J_CF "control_dt10_full")
CA=$(frames $J_CA "control_dt10_adapted_coupled3")
echo "frames: full $FULL | none $NONE | adapted $ADA | control_full $CF | control_adapted $CA"

page() {     # DIPFILE CONTROLFILE NAME
    [ -s "$1" ] && [ -s "$2" ] || { echo "skip $3: missing $1 or $2"; return; }
    python3 plots/plot_state_3d.py "$1" --compare "$2" --x $X0 $X1 --tile 1 --no-temp --no-temperate --no-show \
        && cp "${1%.npz}_3d.html" "$W/$3.html" && echo "wrote $W/$3.html"
}
page "$FULL" "$CF" full_vs_control
page "$NONE" "$CF" none_vs_control
page "$ADA" "$CA" adapted_vs_control
echo
echo "on the Mac:"
echo "rsync -av -e \"ssh -J slloyd@sshproxy-m.geophysik.uni-muenchen.de\" \"slloyd@tethys-3g:$PWD/$W/{full,none,adapted}_vs_control.html\" ~/Desktop/ && open ~/Desktop/{full,none,adapted}_vs_control.html"
