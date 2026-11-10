#!/bin/bash
# Rotational periodic Stokes test (see rotper.sif.in): three runs on one annular sector mesh, then the comparison.
#   bash periodic_test/rotper/run_rotper.sh          (from the repo root; needs the Elmer module and
#                                                     elmer/ESSIncompressibleNS.so: make -C elmer ESSIncompressibleNS.so)
# ANGLE (degrees, default 20), R0, R1, H (m), NP (ranks, default 1: serial; >1: ElmerGrid strips in x, which keep the
# periodic pairs together because the sector is symmetric about the x-axis), DIRECT (default umfpack serial, mumps
# parallel), ROTSIGN (default 1; -1 if Elmer's pairing wants the rotation the other way: "no match" in the log).
# On tethys: sbatch periodic_test/rotper/run_rotper.cmd (module, build, then this).
set -e
cd "$(dirname "$0")"
ANGLE=${ANGLE:-20}; R0=${R0:-10000}; R1=${R1:-60000}; H=${H:-1000}; NP=${NP:-1}; ROTSIGN=${ROTSIGN:-1}
DIRECT=${DIRECT:-$([ "$NP" -gt 1 ] && echo mumps || echo umfpack)}
RANG=$(python3 -c "print($ROTSIGN * $ANGLE)")
LIB=$(cd ../../elmer && pwd)/ESSIncompressibleNS
D=work_a${ANGLE}; mkdir -p $D; cd $D
sed "s/@R0@/$R0/; s/@R1@/$R1/; s/@ANGLE@/$ANGLE/; s/@H@/$H/" ../sector.geo.in > sector.geo
gmsh sector.geo -2 -format msh2 -o sector.msh > gmsh.log
rm -rf sector; ElmerGrid 14 2 sector.msh -autoclean -out sector > elmergrid.log
[ "$NP" -gt 1 ] && ElmerGrid 2 2 sector -partition $NP 1 1 >> elmergrid.log
PER="  Periodic BC = 1
  Periodic BC Rotate(3) = Real 0.0 0.0 $RANG
  Periodic BC Velocity 1 = Logical True
  Periodic BC Velocity 2 = Logical True
  Periodic BC Velocity 3 = Logical True
  Periodic BC Pressure = Logical True"
ROT="  Normal-Tangential Velocity = Logical True
  ESS Rotational Periodic Velocity = Logical True
  Rotational Normals = Logical True"
WALL="  Normal-Tangential Velocity = Logical True
  Velocity 1 = Real 0.0"
for CASE in periodic walls cartesian; do
    case $CASE in
        periodic)  S1=""; S2="$PER
$ROT";;
        walls)     S1="$WALL"; S2="$WALL";;
        cartesian) S1=""; S2="$PER";;
    esac
    python3 - "$CASE" "$LIB" "$R0" "$S1" "$S2" "$DIRECT" <<'PY'
import sys
case, lib, r0, s1, s2, direct = sys.argv[1:]
t = open("../rotper.sif.in").read()
t = t.replace("@CASE@", case).replace("@LIB@", lib).replace("@R0@", r0).replace("@SIDE1@", s1).replace("@SIDE2@", s2)
t = t.replace("@DIRECT@", direct)
open(f"rotper_{case}.sif", "w").write(t)
PY
    echo "$(date +%T) $CASE"
    if [ "$NP" -gt 1 ]; then mpirun -np $NP ElmerSolver_mpi rotper_$CASE.sif > $CASE.log 2>&1 || true
    else ElmerSolver rotper_$CASE.sif > $CASE.log 2>&1 || true; fi
    grep -h "ESS rotational periodic" $CASE.log | tail -1 || true
    grep -hi "error\|fatal\|no match\|could not" $CASE.log | head -5 || true
    grep -q "SOLVER TOTAL TIME" $CASE.log && echo "  finished" || echo "  did not finish: see $D/$CASE.log"
done
cd ..
python3 compare_rotper.py $D $ANGLE
