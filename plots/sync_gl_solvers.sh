#!/bin/bash
# Laptop: fetch the moving-grounding-line solver test from tethys (only the saved states; files gone on the cluster
# are removed here too) and plot it (plots/plot_gl_solvers.py). Run from the repository root:
#   bash plots/sync_gl_solvers.sh                          # direct iterative block bubbles
#   bash plots/sync_gl_solvers.sh direct block             # given runs
# REMOTE (default: the ESS_solvers clone on tethys) and JUMP (the ssh jump host) can be set in the environment.
cd "$(dirname "$0")/.."
RUNS=${*:-direct iterative block bubbles}
REMOTE=${REMOTE:-/SCRATCH/slloyd/github/ElmerStreams/MasterCopy/ElmerStreams/ESS_solvers/work/test_gl_solvers}
JUMP=${JUMP:-slloyd@sshproxy-m.geophysik.uni-muenchen.de}
# one ssh connection for all transfers: the passwords are asked once, and the connection is kept for 4 h (later
# runs within that time ask nothing). End it early with: ssh -O exit -o ControlPath=$CP slloyd@tethys-3g
mkdir -p ~/.ssh
CP=~/.ssh/cm-ess-%r@%h:%p
SSH="ssh -J $JUMP -o ControlMaster=auto -o ControlPath=$CP -o ControlPersist=4h"
$SSH slloyd@tethys-3g true || { echo "could not connect to tethys"; exit 1; }
mkdir -p work/test_gl_solvers
for s in $RUNS; do
    mkdir -p work/test_gl_solvers/$s
    rsync -a --delete -e "$SSH" --include='*/' --include='*.npz' --exclude='*' --prune-empty-dirs \
          slloyd@tethys-3g:$REMOTE/$s/ work/test_gl_solvers/$s/ && echo "$s: synced" || echo "$s: not there (yet)"
done
python plots/plot_gl_solvers.py work/test_gl_solvers $(for s in $RUNS; do [ -n "$(ls work/test_gl_solvers/$s 2>/dev/null)" ] && echo $s; done)
