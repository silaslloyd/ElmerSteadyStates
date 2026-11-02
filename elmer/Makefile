# Build the state I/O solver and the marine sea-pressure functions (needs Elmer's elmerf90 on the PATH,
# e.g. after `module load elmerfem`).
# elmerf90_safe: elmerf90, rerun without unreadable -I/-L directories if it fails on them (elmerfem/2026-10-28)
ELMERF90 ?= ./elmerf90_safe

all: SteadyStateIO.so Buoyancy.so RotateGuess.so GLDistance.so

# make -C elmer checked: the same libraries with bounds and pointer checks (-g -fcheck=all) in elmer/checked/, for
# hunting memory corruption (ESS_ELMER_LIBDIR=$PWD/elmer/checked makes ess use them; jobs/scaling_tests.cmd CHECKED=1)
CHECKFLAGS = -g -O0 -fcheck=all -fbacktrace
checked:
	mkdir -p checked
	$(ELMERF90) $(CHECKFLAGS) -o checked/SteadyStateIO.so SteadyStateIO.F90
	$(ELMERF90) $(CHECKFLAGS) -o checked/Buoyancy.so Buoyancy.F90
	$(ELMERF90) $(CHECKFLAGS) -o checked/RotateGuess.so RotateGuess.F90
	$(ELMERF90) $(CHECKFLAGS) -o checked/PeriodicGL.so PeriodicUtils.F90 FixPeriodicLoads.F90 ContactPeriodic.F90 $(ELMERICE_USF)
	$(ELMERF90) $(CHECKFLAGS) -o checked/ESSFreeSurface.so PeriodicUtils.F90 ESSFreeSurface.F90
	$(ELMERF90) $(CHECKFLAGS) -o checked/StepCheck.so StepCheck.F90

# not in `all` (only the periodic tests need it): make -C elmer PeriodicGL.so
# PeriodicGL.so: PeriodicUtils (periodic copies of nodal fields), the FixPeriodicLoads solver and the periodic
# contact sliding function SlidCoef_ContactPeriodic. The latter calls Elmer/Ice's Sliding_Weertman etc., so it
# is linked against ElmerIceUSF.so (found next to ElmerSolver; override with ELMERICE_USF=...).
ELMER_LIBDIR ?= $(shell d=$$(dirname $$(command -v ElmerSolver 2>/dev/null) 2>/dev/null)/../share/elmersolver/lib; \
                  [ -d "$$d" ] && cd "$$d" && pwd)
ELMERICE_USF ?= $(wildcard $(ELMER_LIBDIR)/ElmerIceUSF.so)

SteadyStateIO.so: SteadyStateIO.F90
	$(ELMERF90) -o $@ $<

Buoyancy.so: Buoyancy.F90
	$(ELMERF90) -o $@ $<

# Before Linsolve procedure: the initial guess rotated to the normal-tangential frame ([elmer] stokes_rotate_guess)
RotateGuess.so: RotateGuess.F90
	$(ELMERF90) -o $@ $<

# the distance to the grounding line, every time step ([elmer] fs_ssa_km: the moving full-Stokes region)
GLDistance.so: GLDistance.F90
	$(ELMERF90) -o $@ $<

PeriodicGL.so: PeriodicUtils.F90 FixPeriodicLoads.F90 ContactPeriodic.F90
	@[ -n "$(ELMERICE_USF)" ] || echo "warning: ElmerIceUSF.so not found; SlidCoef_ContactPeriodic needs it loaded"
	$(ELMERF90) -o $@ $^ $(ELMERICE_USF)

# Elmer's IncompressibleNS (dd6aef5e2) with the "adapted" implicit FSSA ([elmer] fssa = adapted)
ESSIncompressibleNS.so: ESSIncompressibleNS.F90
	$(ELMERF90) -o $@ $<

# the time-step check ([elmer] step_check): CFL, change per step, relaxation number
StepCheck.so: StepCheck.F90
	$(ELMERF90) -o $@ $<

# Elmer's FreeSurfaceSolver with a periodic-aware limiter (Zs, Zb with [case] periodic_sides; see the file's header)
ESSFreeSurface.so: PeriodicUtils.F90 ESSFreeSurface.F90
	$(ELMERF90) -o $@ $^

# ElmerStreams' thermal chain ([elmer] thermal = "elmerstreams") and the no-water sheet (thermal = "nowater"):
# make -C elmer thermal
thermal: ColdTempMask.so SheetSolverhw.so NoWaterSheet.so

NoWaterSheet.so: NoWaterSheet.F90
	$(ELMERF90) -o $@ $<

ColdTempMask.so: ColdTempMaskSolver.F90
	$(ELMERF90) -o $@ $<

SheetSolverhw.so: SheetSolverhw.F90
	$(ELMERF90) -o $@ $<

clean:
	rm -rf checked; rm -f SteadyStateIO.so Buoyancy.so RotateGuess.so GLDistance.so PeriodicGL.so FixPeriodicLoads.so ColdTempMask.so SheetSolverhw.so NoWaterSheet.so ESSFreeSurface.so StepCheck.so ESSIncompressibleNS.so *.mod
