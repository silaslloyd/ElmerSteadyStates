#!/bin/bash
# On tethys: pack the frames of every perturbation run of the land strip (the 100x log-space friction pattern, from the
# 78,000 yr state) into work/land_flowline_3d/cmp_<label>_frames.npz: FSSA full at dt 50 / 100 / 200 / 500 / 1000,
# FSSA none at dt 50 and 10, the implicit (adapted, coupled) FSSA at dt 200 / 500 / 1000. A run that has not started (no run
# folder yet) is skipped; of several jobs with one name the last that was not cancelled is used (cancelled duplicates are ignored); one that is running or has finished is packed with the frames it has.
#   bash plots/fssa_pack.sh              # all
#   bash plots/fssa_pack.sh none imp     # only the runs whose label starts with one of the words (full / none / imp)
cd "$(dirname "$0")/.."
W=work/land_flowline_3d
PRE="$*"; [ -n "$PRE" ] || PRE=""
want() { [ -z "$PRE" ] && return 0; for w in $PRE; do [[ $1 == $w* ]] && return 0; done; return 1; }
for p in "full_dt50 log100-dt50" "full_dt100 log100-dt100" "full_dt200 log100-10k" "full_dt500 log100-dt500" \
         "full_dt1000 log100-dt1000" "none_dt50 log100-none-dt50" "none_dt10 log100-none-dt10" "imp_dt200 log100-imp-dt200" "imp_dt500 log100-imp-dt500" "imp_dt1000 log100-imp-dt1000"; do
    set -- $p
    want $1 || continue
    rm -f $W/cmp_$1_frames.npz
    J=$(sacct -u "$USER" -S 2026-11-25 -n -X -o JobID,JobName%30,State | awk -v n="$2" '$2==n && $3 !~ /CANCELLED/ {j=$1} END{print j}')
    D=$([ -n "$J" ] && ls -d $W/runs/*fssa${J}_* 2>/dev/null | tail -1)
    if [ -z "$D" ]; then echo "$1: no run folder yet (job ${J:-not submitted})"; continue; fi
    echo -n "$1 (job $J): "
    python3 plots/vtu_frames.py $D $W/cmp_$1_frames.npz --t0 78000 2>&1 | tail -1
done
echo; ls -la $W/cmp_*_frames.npz 2>/dev/null
