# Periodic grounding-line test

Does everything stay periodic when the grounding line moves? `run_gl.py`, `gl.sif.in`, `run_gl.cmd`.

**Setup.** A marine slab: divide at x = 0, fixed calving front at x = 64 km (sea pressure), sides y = 0 / y = 4 km
periodic (y = W the image of y = 0). The bed slopes down to the front with a bump across y (40 m, one wavelength
across the domain), so the grounding line is not straight, and the seam forces and the bed normals differ
between the two sides of the seam. Glen ice (n = 3), linear Weertman sliding through the contact sliding
function (the grounding line moves every step, as ElmerStreams), free top and bottom surfaces (Zs, Zb),
accumulation. 64 x 4 cells of 1 km, 8 layers, dt = 1 yr. Forcing in three phases: spin-up (0-150 yr),
retreat (150-300 yr: sliding coefficient x 0.3), advance (300-450 yr: friction back, accumulation 0.5 -> 1.5 m/yr).

**Cases.**
- `old`: ElmerStreams as it is now: the original contact test (`Periodic Contact = False`) on the raw Stokes loads
  (master: the pair total, image: one-sided) and the hydrostatic loads doubled on the master side (`Fw2`, as
  `FixFw`; here `FixPeriodicLoads` with mode `fold`).
- `new`: `SlidCoef_ContactPeriodic` (elmer/ContactPeriodic.F90): the test uses periodic copies of the Stokes loads,
  the hydrostatic loads (`Fw`, undoubled) and the bed normals; the grounded mask is made periodic after the test,
  and again after GroundedSolver (`FixPeriodicLoads`, mode `min`, in place).

**Output.** Per output time (every 10 steps): the grounding line per row across y (the last x with mask >= 0;
the y = 0 and y = W rows are the same physical row), and the largest master/image mismatch of the grounded mask,
Zs, Zb, the velocities, the raw and the periodic Stokes loads, the hydrostatic loads and the normals.
`out_gl/<case>/periodicity.csv` has the same. In `new` everything except the raw loads should match exactly;
in `old` the mask (and through it Zb, the velocities) can differ between the two copies.

**Run.**

    sbatch periodic_test/run_gl.cmd                       # both cases, serial (about an hour each?)
    sbatch periodic_test/run_gl.cmd --steps 30            # a short trial first
    sbatch periodic_test/run_gl.cmd --only new --parts 4  # partitioned (ElmerGrid -periodic 0 1 0)
    python3 periodic_test/run_gl.py --analyse             # the tables again

**Using it in ElmerStreams.** Build `PeriodicGL.so` (`make -C elmer PeriodicGL.so`) and, on the bed BC:

    Slip Coefficient 2 = Variable Coordinate 1
      Real Procedure ".../elmer/PeriodicGL" "SlidCoef_ContactPeriodic"
    Slip Coefficient 3 = Variable Coordinate 1
      Real Procedure ".../elmer/PeriodicGL" "SlidCoef_ContactPeriodic"
    Hydrostatic Loads Variable = String "Fw"        ! not Fw2: drop the FixFw doubling

plus a `FixPeriodicLoads` solver with `Periodic Mode = "min"` on `GroundedMask` after each GroundedSolver, and one
after Stokes for periodic Stokes loads for the friction heating (`frictionloads` without the `/2`).
