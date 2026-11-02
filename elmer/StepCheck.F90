!------------------------------------------------------------------------------
!> StepCheck: per time step, how close the step is to the limits of the free-surface time stepping, logged and
!> written as nodal fields ([elmer] step_check / ESS_STEP_CHECK=1; jobs: STEP_CHECK=1). After Timestep, active on
!> the upper surface (Zs) and the ice base (Zb) boundary elements. h is the shortest horizontal edge of the
!> boundary elements around a node (plan view).
!>
!>   advective CFL      |u_h| dt / h            the surface and the floating base (the Zs / Zb equations advect
!>                                              with the horizontal velocity; SUPG is stable above 1, but the
!>                                              surface is then moved further than one element per step)
!>   step change        |Zs - Zs_prev|, |Zb - Zb_prev| (floating base) per step, and the largest as a fraction of
!>                                              the ice thickness Zs - Zb
!>   relaxation number  R = dt rho_i g h / (2 pi mu)
!>                      the shortest resolved surface wave (wavelength 2h) on viscous ice relaxes at the rate
!>                      gamma = rho_i g / (2 mu k), k = pi / h (thick-ice limit, an upper bound on gamma; thinner
!>                      ice relaxes slower). An explicit (forward) free-surface update is stable only for
!>                      dt gamma < 2, i.e. R < 2; fully implicit (FSSA theta = 1) has no such limit. Linear ice
!>                      (Viscosity = mu); with Glen's law it is a rough estimate from the given mu.
!> The maxima are reduced over all ranks and logged with their position (Info, level 3); the nodal values go to
!> the exported variables "CFL" (advective) and "Step Change" (|dZs| on the surface, |dZb| on the floating base).
!>
!> Keywords (Solver section):  Variable = "CFL";  Exported Variable 1 = -dofs 1 "Step Change"
!>   Ice Density = Real, Gravity = Real, Viscosity = Real    (the model's units)
!------------------------------------------------------------------------------
SUBROUTINE StepCheck( Model, Solver, dt, Transient )
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
  TYPE(Element_t), POINTER :: Element
  TYPE(Variable_t), POINTER :: cVar, sVar, u1, u2, zsVar, zbVar, gmVar, tVar
  REAL(KIND=dp), ALLOCATABLE :: h(:), zsN(:), zbN(:)
  INTEGER, ALLOCATABLE :: zsOf(:)
  REAL(KIND=dp) :: rhoi, grav, mu, uh, c, dz, hmin, e, thick, &
      cTop, cBot, dTop, dBot, rFrac, rMax, xs(3), ys(3), loc(5,2), nTop1, nBot1, tnow
  INTEGER :: t, i, j, k, n, a, b, nn, p
  LOGICAL :: Found, top, bot, GotR
  CHARACTER(*), PARAMETER :: Caller = 'StepCheck'
!------------------------------------------------------------------------------
  Params => GetSolverParams()
  Mesh => GetMesh()
  nn = Mesh % NumberOfNodes
  cVar => Solver % Variable
  sVar => VariableGet( Mesh % Variables, 'Step Change', ThisOnly = .TRUE. )
  u1 => VariableGet( Mesh % Variables, 'Velocity 1' )
  u2 => VariableGet( Mesh % Variables, 'Velocity 2' )
  zsVar => VariableGet( Mesh % Variables, 'Zs' )
  zbVar => VariableGet( Mesh % Variables, 'Zb' )
  gmVar => VariableGet( Mesh % Variables, 'GroundedMask' )
  IF ( .NOT. ( ASSOCIATED(u1) .AND. ASSOCIATED(u2) .AND. ASSOCIATED(zsVar) ) ) &
      CALL Fatal( Caller, 'needs Velocity 1, Velocity 2 and Zs' )
  rhoi = ListGetConstReal( Params, 'Ice Density', GotR )
  grav = ListGetConstReal( Params, 'Gravity', Found ); GotR = GotR .AND. Found
  mu = ListGetConstReal( Params, 'Viscosity', Found ); GotR = GotR .AND. Found

  ! h: the shortest horizontal edge of the active boundary elements at each node
  ALLOCATE( h(nn) )
  h = HUGE(1.0_dp)
  DO t = 1, Solver % NumberOfActiveElements
    Element => GetActiveElement(t)
    n = GetElementNOFNodes( Element )
    IF ( n < 2 ) CYCLE
    hmin = HUGE(1.0_dp)
    DO a = 1, MIN( n, 4 )
      b = MOD( a, MIN( n, 4 ) ) + 1
      i = Element % NodeIndexes(a); j = Element % NodeIndexes(b)
      e = SQRT( ( Mesh % Nodes % x(i) - Mesh % Nodes % x(j) )**2 + ( Mesh % Nodes % y(i) - Mesh % Nodes % y(j) )**2 )
      IF ( e > 0.0_dp ) hmin = MIN( hmin, e )
    END DO
    DO a = 1, n
      i = Element % NodeIndexes(a)
      h(i) = MIN( h(i), hmin )
    END DO
  END DO

  ! the ice thickness at a base node: Zs of the surface node in the same column (same x, y)
  ALLOCATE( zsN(nn), zbN(nn), zsOf(nn) )
  zsN = 0.0_dp; zbN = 0.0_dp
  CALL ColumnZs()

  cTop = 0; cBot = 0; dTop = 0; dBot = 0; rFrac = 0; rMax = 0; nTop1 = 0; nBot1 = 0
  loc = -HUGE(1.0_dp)
  DO i = 1, nn
    IF ( h(i) >= HUGE(1.0_dp) ) CYCLE
    p = 0
    IF ( ASSOCIATED( cVar % Perm ) ) p = cVar % Perm(i)
    top = ( pval( zsVar, i ) /= 0 )
    bot = .FALSE.
    IF ( .NOT. top .AND. ASSOCIATED( zbVar ) ) bot = ( pval( zbVar, i ) /= 0 )
    IF ( bot .AND. ASSOCIATED( gmVar ) ) bot = ( nval( gmVar, i ) < -0.5_dp )   ! the floating base only
    uh = SQRT( nval( u1, i )**2 + nval( u2, i )**2 )
    c = uh * dt / h(i)
    dz = 0.0_dp
    IF ( top ) THEN
      dz = ABS( nval( zsVar, i ) - prevval( zsVar, i ) )
      IF ( c > 1.0_dp .AND. own(i) ) nTop1 = nTop1 + 1
      IF ( c > cTop ) THEN; cTop = c; loc(1,1) = Mesh % Nodes % x(i); loc(1,2) = Mesh % Nodes % y(i); END IF
      IF ( dz > dTop ) THEN; dTop = dz; loc(2,1) = Mesh % Nodes % x(i); loc(2,2) = Mesh % Nodes % y(i); END IF
    ELSE IF ( bot ) THEN
      dz = ABS( nval( zbVar, i ) - prevval( zbVar, i ) )
      IF ( c > 1.0_dp .AND. own(i) ) nBot1 = nBot1 + 1
      IF ( c > cBot ) THEN; cBot = c; loc(3,1) = Mesh % Nodes % x(i); loc(3,2) = Mesh % Nodes % y(i); END IF
      IF ( dz > dBot ) THEN; dBot = dz; loc(4,1) = Mesh % Nodes % x(i); loc(4,2) = Mesh % Nodes % y(i); END IF
      IF ( zsOf(i) > 0 ) THEN
        thick = zsN( zsOf(i) ) - nval( zbVar, i )
        IF ( thick > 1.0_dp ) THEN
          IF ( dz / thick > rFrac ) THEN; rFrac = dz / thick; loc(5,1) = Mesh % Nodes % x(i); loc(5,2) = Mesh % Nodes % y(i); END IF
        END IF
      END IF
    ELSE
      c = 0.0_dp
    END IF
    IF ( GotR .AND. ( top .OR. bot ) ) rMax = MAX( rMax, dt * rhoi * grav * h(i) / ( 2.0_dp * PI * mu ) )
    IF ( p > 0 ) cVar % Values(p) = c
    IF ( ASSOCIATED( sVar ) ) THEN
      IF ( ASSOCIATED( sVar % Perm ) ) THEN
        IF ( sVar % Perm(i) > 0 ) sVar % Values( sVar % Perm(i) ) = dz
      END IF
    END IF
  END DO

  ! maxima over all ranks, each with the position of the rank that has it
  CALL GMax( cTop, 1 ); CALL GMax( dTop, 2 ); CALL GMax( cBot, 3 ); CALL GMax( dBot, 4 ); CALL GMax( rFrac, 5 )
  IF ( ParEnv % PEs > 1 ) THEN
    rMax = ParallelReduction( rMax, 2 )
    nTop1 = ParallelReduction( nTop1 ); nBot1 = ParallelReduction( nBot1 )
  END IF
  tnow = 0.0_dp
  tVar => VariableGet( Mesh % Variables, 'Time' )
  IF ( ASSOCIATED( tVar ) ) tnow = tVar % Values(1)
  WRITE( Message, '(A,F10.3,A,F9.3,A,F8.2,A,F7.2,A,I0,A,F9.3,A,F8.2,A,F7.2,A,I0,A)' ) 'step check t', tnow, &
      ': CFL surface ', cTop, ' (x', loc(1,1)/1e3_dp, ' y', loc(1,2)/1e3_dp, ' km; ', NINT(nTop1), ' nodes > 1), floating base ', &
      cBot, ' (x', loc(3,1)/1e3_dp, ' y', loc(3,2)/1e3_dp, ' km; ', NINT(nBot1), ' nodes > 1)'
  CALL Info( Caller, Message, Level=3 )
  WRITE( Message, '(A,F10.3,A,F9.3,A,F8.2,A,F7.2,A,F9.3,A,F8.2,A,F7.2,A,ES9.2,A,F8.2,A,F7.2,A,F8.3,A)' ) &
      'step check t', tnow, ': max |dZs| ', dTop, ' m (x', loc(2,1)/1e3_dp, ' y', loc(2,2)/1e3_dp, &
      '), max |dZb| floating ', dBot, ' m (x', loc(4,1)/1e3_dp, ' y', loc(4,2)/1e3_dp, '), max |dZb|/H ', rFrac, &
      ' (x', loc(5,1)/1e3_dp, ' y', loc(5,2)/1e3_dp, '), relaxation number R ', rMax, ' (explicit stable < 2)'
  CALL Info( Caller, Message, Level=3 )
  DEALLOCATE( h, zsN, zbN, zsOf )

CONTAINS

  SUBROUTINE GMax( v, k )
    REAL(KIND=dp) :: v
    INTEGER :: k
    REAL(KIND=dp) :: vl
    IF ( ParEnv % PEs <= 1 ) RETURN
    vl = v
    v = ParallelReduction( v, 2 )
    IF ( vl < v ) loc(k,:) = -HUGE(1.0_dp)
    loc(k,1) = ParallelReduction( loc(k,1), 2 )
    loc(k,2) = ParallelReduction( loc(k,2), 2 )
  END SUBROUTINE GMax

  SUBROUTINE ColumnZs()
    ! surface nodes keyed by (x, y) rounded to 0.1 m: a sort of the surface nodes, then a binary search per base node
    INTEGER :: ns, m, lo, hi, mid, gap, q, r, tmp
    INTEGER, ALLOCATABLE :: ids(:), ord(:)
    REAL(KIND=dp), ALLOCATABLE :: key(:)
    zsOf = 0
    ns = 0
    DO m = 1, nn
      IF ( h(m) < HUGE(1.0_dp) .AND. pval( zsVar, m ) /= 0 ) ns = ns + 1
    END DO
    IF ( ns == 0 ) RETURN
    ALLOCATE( ids(ns), ord(ns), key(ns) )
    ns = 0
    DO m = 1, nn
      IF ( h(m) < HUGE(1.0_dp) .AND. pval( zsVar, m ) /= 0 ) THEN
        ns = ns + 1; ids(ns) = m; key(ns) = ckey( m ); zsN(m) = nval( zsVar, m )
      END IF
    END DO
    DO m = 1, ns
      ord(m) = m
    END DO
    ! shell sort of ord by key (inline: an internal procedure cannot contain another)
    gap = ns / 2
    DO WHILE ( gap > 0 )
      DO q = gap + 1, ns
        tmp = ord(q); r = q
        DO WHILE ( r > gap )
          IF ( key( ord(r-gap) ) <= key(tmp) ) EXIT
          ord(r) = ord(r-gap); r = r - gap
        END DO
        ord(r) = tmp
      END DO
      gap = gap / 2
    END DO
    DO m = 1, nn
      IF ( h(m) >= HUGE(1.0_dp) .OR. pval( zsVar, m ) /= 0 ) CYCLE
      lo = 1; hi = ns
      DO WHILE ( lo < hi )
        mid = ( lo + hi ) / 2
        IF ( key( ord(mid) ) < ckey(m) ) THEN; lo = mid + 1; ELSE; hi = mid; END IF
      END DO
      IF ( ABS( key( ord(lo) ) - ckey(m) ) < 0.5_dp ) zsOf(m) = ids( ord(lo) )
    END DO
    DEALLOCATE( ids, ord, key )
  END SUBROUTINE ColumnZs

  FUNCTION ckey( m ) RESULT( kk )      ! a scalar key of the plan position (x, y) at 0.1 m: y < 1e6 m
    INTEGER :: m
    REAL(KIND=dp) :: kk
    kk = ANINT( Mesh % Nodes % x(m) * 10.0_dp ) * 1.0e7_dp + ANINT( Mesh % Nodes % y(m) * 10.0_dp )
  END FUNCTION ckey

  FUNCTION pval( v, m ) RESULT( q )    ! the variable's index of node m (0: not there)
    TYPE(Variable_t), POINTER :: v
    INTEGER :: m, q
    q = 0
    IF ( .NOT. ASSOCIATED( v ) ) RETURN
    q = m
    IF ( ASSOCIATED( v % Perm ) ) THEN
      q = 0
      IF ( m <= SIZE( v % Perm ) ) q = v % Perm(m)
    END IF
  END FUNCTION pval

  FUNCTION nval( v, m ) RESULT( val )
    TYPE(Variable_t), POINTER :: v
    INTEGER :: m
    REAL(KIND=dp) :: val
    INTEGER :: q
    val = 0.0_dp
    q = pval( v, m )
    IF ( q <= 0 .OR. q > SIZE( v % Values ) ) RETURN
    val = v % Values(q)
  END FUNCTION nval

  FUNCTION prevval( v, m ) RESULT( val )   ! the value at the start of this time step
    TYPE(Variable_t), POINTER :: v
    INTEGER :: m
    REAL(KIND=dp) :: val
    INTEGER :: q
    val = nval( v, m )
    IF ( .NOT. ASSOCIATED( v % PrevValues ) ) RETURN
    q = pval( v, m )
    IF ( q <= 0 .OR. q > SIZE( v % PrevValues, 1 ) ) RETURN
    val = v % PrevValues(q, 1)
  END FUNCTION prevval

  FUNCTION own( m ) RESULT( yes )
    INTEGER :: m
    LOGICAL :: yes
    yes = .TRUE.
    IF ( ParEnv % PEs > 1 ) yes = ( Mesh % ParallelInfo % NeighbourList(m) % Neighbours(1) == ParEnv % MyPE )
  END FUNCTION own
!------------------------------------------------------------------------------
END SUBROUTINE StepCheck
