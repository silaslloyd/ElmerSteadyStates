# Build the state I/O solver and the marine sea-pressure functions (needs Elmer's elmerf90 on the PATH,
# e.g. after `module load elmerfem`).
ELMERF90 ?= elmerf90

all: SteadyStateIO.so Buoyancy.so

# not in `all`: only the periodic loads test needs it (make -C elmer FixPeriodicLoads.so)

SteadyStateIO.so: SteadyStateIO.F90
	$(ELMERF90) -o $@ $<

Buoyancy.so: Buoyancy.F90
	$(ELMERF90) -o $@ $<

FixPeriodicLoads.so: FixPeriodicLoads.F90
	$(ELMERF90) -o $@ $<

clean:
	rm -f SteadyStateIO.so Buoyancy.so FixPeriodicLoads.so *.mod
