#!/bin/bash
#SBATCH --output=./slurms/flowline_bounds_%j.out
#SBATCH --job-name=flbounds
#SBATCH --ntasks=128
#SBATCH --time=3-00:00:00
#
# The two bounding steady states of the extended strip's flowline (configs/flowline_rect_ext.toml), mechanics only
# (Stokes and the free surfaces, no heat: neither bed's sliding depends on temperature):
#   cold       the grounded bed frozen to its substrate, no sliding (ESS_SLIDING=frozen)   -> bound_cold.npz
#   temperate  the grounded bed sliding as at the melting point everywhere (ESS_SLIDING=temperate) -> bound_temperate.npz
# Each: a short mechanics-only spin-up by time stepping from START (default the spin-up of jobs/flowline_steady.cmd,
# work/flowline_rect_ext/spin.npz; SPIN_CHUNKS x 100 x 10 yr, default 10), then Newton on the geometry (ess newton
# --stokes-only; the grounding line moved between solves).
# Same ranks as the mesh's partitioning (128). Steps skipped when their output exists. WHICH="cold temperate" (default
# both, in that order). SPIN_DT (yr, default 10); NEWTON=0: time stepping only (the spin-up's state is the bound; its
# "rms change" per chunk in the log shows how steady it got). SPIN_LINEAR (default iterative: GCR + ILU, several times
# faster than MUMPS here; the spin-up needs no exact map) is the spin-up's Stokes linear solver; Newton keeps the
# config's direct solver. SPIN_GL (default contact): the grounding line moves every step as in the 3D transient runs
# (ElmerStreams contact; the cold bed then frozen by friction, fcoeff x ess_slide_maxfactor, instead of no slip);
# frozen: the grounded mask moved by flotation between chunks only (as Newton; the cold bed no-slip).
# FROZEN_FACTOR: the cold bed's friction factor with contact (default ess_slide_maxfactor, 1e6; GCR + ILU stalls there,
# so with 1e6 the cold bound needs SPIN_LINEAR_COLD=direct). SPIN_LINEAR_COLD: the cold bound's Stokes solver
# (default SPIN_LINEAR).
#   sbatch jobs/flowline_bounds.cmd
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurms
export ELMER_MODULE=${ELMER_MODULE:-elmerfem/2026-10-28} ELMER_PREFIX=${ELMER_PREFIX:-/SCRATCH/$USER/elmer/install2}
bash periodic_test/run_gl_ess.cmd build
module load "$ELMER_MODULE"
export PATH=$ELMER_PREFIX/bin:$PATH ELMER_HOME=$ELMER_PREFIX
export ELMER_LIB=$ELMER_PREFIX/share/elmersolver/lib ELMER_SOLVER_HOME=$ELMER_PREFIX/share/elmersolver
export LD_LIBRARY_PATH=$ELMER_PREFIX/lib:$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/share/elmersolver/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ELMER_PREFIX/lib/elmersolver:$ELMER_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export ESS_MPIRUN="mpirun -np {n} --bind-to none"
CFG=configs/flowline_rect_ext.toml
W=work/flowline_rect_ext
START=${START:-$W/spin.npz}
for which in ${WHICH:-cold temperate}; do
    case $which in cold) sl=frozen;; temperate) sl=temperate;; *) echo "WHICH: cold | temperate"; exit 1;; esac
    export ESS_SLIDING=$sl
    lin=${SPIN_LINEAR:-iterative}
    if [ $which = cold ]; then
        lin=${SPIN_LINEAR_COLD:-$lin}
        [ -n "$FROZEN_FACTOR" ] && export ESS_FROZEN_FACTOR=$FROZEN_FACTOR
    else
        unset ESS_FROZEN_FACTOR
    fi
    if [ ! -f $W/bound_${which}_spin.npz ]; then
        echo "$(date +%T) $which bound: mechanics-only spin-up from $START (sliding $sl, grounding line ${SPIN_GL:-contact}), ${SPIN_CHUNKS:-10} x 100 x ${SPIN_DT:-10} yr, Stokes $lin${ESS_FROZEN_FACTOR:+, frozen friction factor $ESS_FROZEN_FACTOR}"
        python3 -u -m ess run $CFG --state $START --stokes-only --gl ${SPIN_GL:-contact} --chunks ${SPIN_CHUNKS:-10} --steps 100 \
            --dt ${SPIN_DT:-10} --linear-stokes $lin --out $W/bound_${which}_spin.npz
    fi
    if [ "${NEWTON:-1}" = 1 ] && [ ! -f $W/bound_${which}.npz ]; then
        echo "$(date +%T) $which bound: Newton on the geometry"
        python3 -u -m ess newton $CFG --state $W/bound_${which}_spin.npz --stokes-only --out $W/bound_${which}.npz || \
            echo "$(date +%T) $which bound: Newton did not converge (see the log)"
    fi
done
echo "$(date +%T) done"
