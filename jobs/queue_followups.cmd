#!/bin/bash
#SBATCH --output=./slurms/followups_%j.out
#SBATCH --job-name=flfollow
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#
# A one-core waiter that submits the follow-up jobs by itself when the flowline pipeline reaches them:
#   * S1 done (pipeline/flags/s1_done): the grounding-line mesh branches, RES=250 (control) and RES=125
#     (jobs/gl_branch.cmd; RESLIST="250 125" to change);
#   * S3 started (pipeline/start_temperate.npz exists, so S2's Newton state is final): the 3D free-surface test: fssa3d/prep.cmd
#     from the remeshed config and the S2 state, then fssa3d/sweep.sh behind it (--dependency=afterok).
#   sbatch jobs/queue_followups.cmd
# CONFIG (default configs/flowline_gl_fine.toml), RESLIST, SWEEP_ENV (e.g. "DT=10 YEARS=1000 DIP=0.5 NT=64"),
# BRANCHES=0 / FSSA3D=0 to leave one out. Polls every 2 minutes; the jobs it submits are listed in slurms/followups_<id>.out.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CONFIG=${CONFIG:-configs/flowline_gl_fine.toml}
WORK=$(python3 -c "import os; from ess.config import load_config as L; print(os.path.relpath(L('$CONFIG')['_work']))")
P=$WORK/pipeline
echo "$(date +%T) waiting for $P/flags/s1_done"
if [ "${BRANCHES:-1}" != 0 ]; then
    until [ -f $P/flags/s1_done ]; do sleep 120; done
    for r in ${RESLIST:-250 125}; do
        echo "$(date +%T) S1 done: submitting the $r m branch: $(RES=$r CONFIG=$CONFIG sbatch --parsable jobs/gl_branch.cmd)"
    done
fi
if [ "${FSSA3D:-1}" != 0 ]; then
    echo "$(date +%T) waiting for $P/start_temperate.npz (S3 started: S2 final)"
    until [ -f $P/start_temperate.npz ]; do sleep 120; done
    SRC=$P/stokes_newton.npz; [ -f $SRC ] || SRC=$P/start_cut.npz       # S2 did not converge: the transient's cut state
    PREP=$(SRC_CFG=$WORK/refined/pipe1.toml SRC_STATE=$SRC sbatch --parsable fssa3d/prep.cmd)
    echo "$(date +%T) 3D prep $PREP from $SRC; the sweep follows it:"
    env ${SWEEP_ENV:-} DEP=$PREP bash fssa3d/sweep.sh
fi
echo "$(date +%T) all follow-ups submitted"
