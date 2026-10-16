# Build the state I/O solver (needs Elmer's elmerf90 on the PATH, e.g. after `module load elmerfem`).
ELMERF90 ?= elmerf90

SteadyStateIO.so: SteadyStateIO.F90
	$(ELMERF90) -o $@ $<

clean:
	rm -f SteadyStateIO.so *.mod
