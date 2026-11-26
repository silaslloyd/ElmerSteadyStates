#!/bin/bash
# The 8-iteration implicit land run (reference job log100-impt-dt500) twice, BDF 1 and BDF 2, with the SAME fixed Stokes
# preconditioner (default ILU3, as yesterday's runs), tolerance 1e-4, cap 8, log100 friction, output as the reference.
#   DRY=1 bash jobs/land3d_log100_bdf12.sh        # print the two sbatch lines only
#   bash jobs/land3d_log100_bdf12.sh              # submit both
#   STOKES_PREC=ILU4 bash jobs/land3d_log100_bdf12.sh     # another preconditioner for both
cd "$(dirname "$0")/.."
export REFNAME=${REFNAME:-log100-impt-dt500} SS_TOL=${SS_TOL:-1e-4} STOKES_PREC=${STOKES_PREC:-ILU3}
export START=${START:-work/land_flowline_3d/state_078000.npz}
for B in 1 2; do
    echo "=== BDF $B"
    BDF=$B bash jobs/land3d_log100_bdf2.sh
done
