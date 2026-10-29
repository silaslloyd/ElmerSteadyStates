!------------------------------------------------------------------------------
! RotateInitialGuess: a "Before Linsolve" procedure for solvers with normal-tangential boundary conditions.
!
! Elmer solves such systems in the rotated (normal-tangential) frame at the boundary nodes and rotates the
! solution back to x/y/z afterwards, but it passes the current solution to the iterative solver as the initial
! guess WITHOUT rotating it first (only "Linear System Residual Mode" does; SolveCore.F90, SolveSystem). At
! every normal-tangential node the guess then has its components mixed up: a warm start (the previous time
! step, the previous nonlinear iteration) begins with a residual far above that of a zero guess (ESS: 60 x the
! right-hand side norm instead of < 1) and loses everything it should have saved.
!
! This procedure rotates the guess into the frame of the system, exactly as residual mode does, just before
! the linear solve; Elmer back-rotates the solution as usual. Use in the solver section:
!   Before Linsolve = File "RotateGuess" "RotateInitialGuess"
! Only the starting point of the iteration changes, not the system or its solution. Monolithic solves only
! (not the block solver).
!------------------------------------------------------------------------------
INTEGER FUNCTION RotateInitialGuess( Model, Solver, A, b, x, n, DOFs, Norm )
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t)  :: Model
  TYPE(Solver_t) :: Solver
  TYPE(Matrix_t), POINTER :: A
  INTEGER :: n, DOFs
  REAL(KIND=dp) :: b(n), x(n), Norm
  INTEGER, POINTER :: Perm(:)
  LOGICAL, SAVE :: Reported = .FALSE.

  RotateInitialGuess = 0          ! nonzero would make Elmer skip the solve
  IF ( .NOT. ASSOCIATED( Solver % Variable % Perm ) ) RETURN
  IF ( Solver % NormalTangential % NormalTangentialNOFNodes <= 0 ) RETURN
  Perm => Solver % Variable % Perm
  CALL RotateNTSystemAll( x, Perm, DOFs )
  IF ( .NOT. Reported ) THEN
    CALL Info( 'RotateInitialGuess', 'initial guess rotated to the normal-tangential frame at '// &
        I2S(Solver % NormalTangential % NormalTangentialNOFNodes)//' nodes', Level=3 )
    Reported = .TRUE.
  END IF
END FUNCTION RotateInitialGuess
