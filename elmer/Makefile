# Build the state I/O solver and the marine sea-pressure functions (needs Elmer's elmerf90 on the PATH,
# e.g. after `module load elmerfem`).
ELMERF90 ?= elmerf90

all: SteadyStateIO.so Buoyancy.so

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

PeriodicGL.so: PeriodicUtils.F90 FixPeriodicLoads.F90 ContactPeriodic.F90
	@[ -n "$(ELMERICE_USF)" ] || echo "warning: ElmerIceUSF.so not found; SlidCoef_ContactPeriodic needs it loaded"
	$(ELMERF90) -o $@ $^ $(ELMERICE_USF)

clean:
	rm -f SteadyStateIO.so Buoyancy.so PeriodicGL.so FixPeriodicLoads.so *.mod
