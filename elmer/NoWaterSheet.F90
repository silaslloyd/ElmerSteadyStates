!------------------------------------------------------------------------------
!> NoWaterSheet: the "no water" limit of the ElmerStreams water sheet, for the cold/temperate mask
!> (ColdTempMaskSolver) without a hydrology model ([elmer] thermal = "nowater" in ESS).
!>
!> No storage and no transport: the sheet is only the sign of the basal melt rate,
!>     hw = sign(melt)  on grounded temperate nodes (ColdTempMask > 0),
!>     hw = 0           on grounded cold nodes (their switch back to temperate goes by temperature only),
!>     hw = 1           under floating ice (water everywhere there: a node that grounds starts temperate).
!> ColdTempMaskSolver (Before Timestep) then turns a temperate node cold at the next step when hw <= 0, i.e. as
!> soon as its heat balance at the melting point is a freezing one. Equivalent to an FE sheet with identity
!> stiffness and the nodal sign(melt) as the force (no mass matrix), without assembling or solving anything.
!>
!> The melt rate (ElmerStreams getmeltrate) is -L / (w rho_i L_w): L the heat solver's nodal load at the
!> temperate (T = T0 Dirichlet) node, w its boundary weight. Only its sign is used here; the magnitude is written
!> to the exported variable "Melt Rate" (ice equivalent, m/a) for output only, where w > 0.
!>
!> Run After Timestep, after the periodic copies of the loads (FixPeriodicLoads), on the bed (interface) body.
!> Keywords (Solver section):
!>   Variable = "Water Sheet Thickness"
!>   Loads Variable   = String "Periodic Temperature Loads"   (default; "Temperature Loads" without periodic sides)
!>   Weights Variable = String "Periodic Temperature Boundary Weights"   (optional: for the melt rate output)
!>   Ice Density = Real, Latent Heat = Real                    (for the melt rate output; the model's units)
!>   Exported Variable 1 = -dofs 1 "Melt Rate"
!------------------------------------------------------------------------------
SUBROUTINE NoWaterSheet( Model, Solver, dt, Transient )
!------------------------------------------------------------------------------
  USE DefUtils
  USE ParallelUtils, ONLY : ParallelReduction
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t) :: Solver
  REAL(KIND=dp) :: dt
  LOGICAL :: Transient
!------------------------------------------------------------------------------
  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(ValueList_t), POINTER :: Params
  TYPE(Variable_t), POINTER :: hwVar, ctVar, gmVar, lVar, wVar, mVar
  CHARACTER(LEN=MAX_NAME_LEN) :: lname, wname
  CHARACTER(*), PARAMETER :: Caller = 'NoWaterSheet'
  REAL(KIND=dp) :: rhoi, Lw, gm, ct, ld, w, hw, melt, cnt(5)
  INTEGER :: i, p
  LOGICAL :: Found, GotW
!------------------------------------------------------------------------------
  Params => GetSolverParams()
  Mesh => GetMesh()
  hwVar => Solver % Variable
  IF ( .NOT. ASSOCIATED( hwVar ) ) CALL Fatal( Caller, 'no solver variable ("Water Sheet Thickness")' )

  lname = ListGetString( Params, 'Loads Variable', Found )
  IF ( .NOT. Found ) lname = 'Periodic Temperature Loads'
  lVar => VariableGet( Mesh % Variables, TRIM(lname), ThisOnly = .TRUE. )
  IF ( .NOT. ASSOCIATED( lVar ) ) CALL Fatal( Caller, 'no variable "'//TRIM(lname)//'" (heat solver: Calculate Loads)' )
  ctVar => VariableGet( Mesh % Variables, 'ColdTempMask', ThisOnly = .TRUE. )
  IF ( .NOT. ASSOCIATED( ctVar ) ) CALL Fatal( Caller, 'no variable "ColdTempMask"' )
  gmVar => VariableGet( Mesh % Variables, 'GroundedMask', ThisOnly = .TRUE. )
  IF ( .NOT. ASSOCIATED( gmVar ) ) CALL Fatal( Caller, 'no variable "GroundedMask"' )

  wname = ListGetString( Params, 'Weights Variable', GotW )
  NULLIFY( wVar )
  IF ( GotW ) wVar => VariableGet( Mesh % Variables, TRIM(wname), ThisOnly = .TRUE. )
  GotW = ASSOCIATED( wVar )
  mVar => VariableGet( Mesh % Variables, 'Melt Rate', ThisOnly = .TRUE. )
  rhoi = ListGetConstReal( Params, 'Ice Density', Found )
  IF ( .NOT. Found ) GotW = .FALSE.
  Lw = ListGetConstReal( Params, 'Latent Heat', Found )
  IF ( .NOT. Found ) GotW = .FALSE.

  ! counts (owned nodes): temperate melting, temperate freezing (-> cold next step), cold, floating, zero load
  cnt = 0.0_dp
  DO i = 1, Mesh % NumberOfNodes
    p = hwVar % Perm(i)
    IF ( p <= 0 ) CYCLE
    gm = nval( gmVar, i )
    ct = nval( ctVar, i )
    melt = 0.0_dp
    IF ( gm < -0.5_dp ) THEN
      hw = 1.0_dp
      IF ( own(i) ) cnt(4) = cnt(4) + 1
    ELSE IF ( ct > 0.0_dp ) THEN
      ld = nval( lVar, i )
      hw = SIGN( 1.0_dp, -ld )
      IF ( ld == 0.0_dp ) hw = 0.0_dp
      IF ( own(i) ) THEN
        IF ( hw > 0.0_dp ) cnt(1) = cnt(1) + 1
        IF ( hw < 0.0_dp ) cnt(2) = cnt(2) + 1
        IF ( hw == 0.0_dp ) cnt(5) = cnt(5) + 1
      END IF
      IF ( GotW ) THEN
        w = nval( wVar, i )
        IF ( w > 0.0_dp .AND. w < HUGE(w) ) melt = -ld / ( w * rhoi * Lw )
      END IF
    ELSE
      hw = 0.0_dp
      IF ( own(i) ) cnt(3) = cnt(3) + 1
    END IF
    hwVar % Values(p) = hw
    IF ( ASSOCIATED( mVar ) ) THEN
      IF ( mVar % Perm(i) > 0 ) mVar % Values( mVar % Perm(i) ) = melt
    END IF
  END DO
  IF ( ParEnv % PEs > 1 ) THEN
    DO i = 1, 5
      cnt(i) = ParallelReduction( cnt(i) )
    END DO
  END IF
  WRITE( Message, '(A,I0,A,I0,A,I0,A,I0,A,I0)' ) 'bed nodes: temperate melting ', NINT(cnt(1)), &
      ', temperate freezing (cold next step) ', NINT(cnt(2)), ', cold ', NINT(cnt(3)), ', floating ', NINT(cnt(4)), &
      ', temperate zero load ', NINT(cnt(5))
  CALL Info( Caller, Message, Level=3 )

CONTAINS

  FUNCTION nval( v, k ) RESULT( val )
    TYPE(Variable_t), POINTER :: v
    INTEGER :: k
    REAL(KIND=dp) :: val
    INTEGER :: q
    val = 0.0_dp
    q = k
    IF ( ASSOCIATED( v % Perm ) ) THEN
      IF ( k > SIZE( v % Perm ) ) RETURN
      q = v % Perm(k)
    END IF
    IF ( q <= 0 .OR. q > SIZE( v % Values ) ) RETURN
    val = v % Values(q)
  END FUNCTION nval

  FUNCTION own( k ) RESULT( yes )
    INTEGER :: k
    LOGICAL :: yes
    yes = .TRUE.
    IF ( ParEnv % PEs > 1 ) yes = ( Mesh % ParallelInfo % NeighbourList(k) % Neighbours(1) == ParEnv % MyPE )
  END FUNCTION own
!------------------------------------------------------------------------------
END SUBROUTINE NoWaterSheet
