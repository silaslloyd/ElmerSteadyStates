!------------------------------------------------------------------------------
!> RemeshTrigger: ends the simulation when the mesh no longer resolves where the grounding line and the cold/temperate
!> transition are (`ess run --remesh-auto`: the driver then remeshes from the final state and carries on).
!> After Timestep, active on the ice base ("interface equations").
!>
!> "Remesh Distance" (exported, read from the state: written by `ess remesh` for the mesh it made) is, at every base
!> node, the distance (m) to the nearest element coarser than the fine size (1.3 x --fine), i.e. how far a curve can
!> move from that node before it runs into coarse mesh. Each step:
!>   grounding line   the nodes with |GroundedMask| < 0.5: remesh when any has Remesh Distance < "GL Safety Distance"
!>   transition       the grounded base nodes in a boundary element holding both a cold (ColdTempMask <= 0) and a
!>                    temperate node: remesh when more than "Transition Fraction" of them have Remesh Distance <
!>                    "Transition Safety Distance" (a few close ones: new temperate patches inland, not a margin
!>                    leaving its band)
!> Then (all ranks agree, by reductions) the Simulation keyword "Exit Condition" is set to 1, so ElmerSolver leaves
!> the time loop after this step (the After All state write still runs), and rank 0 writes the file "remesh_needed"
!> (time, step, the counts) in the run directory for the driver. Not before "Minimum Steps" steps of this run.
!>
!> Keywords (Solver section): Exported Variable 1 = -dofs 1 "Remesh Distance"
!>   GL Safety Distance = Real (m, default 5000), Transition Safety Distance = Real (m, default 1000),
!>   Transition Fraction = Real (default 0.05), Minimum Steps = Integer (default 1)
!------------------------------------------------------------------------------
SUBROUTINE RemeshTrigger( Model, Solver, dt, Transient )
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
  TYPE(Variable_t), POINTER :: dVar, gmVar, ctVar, tVar
  LOGICAL, ALLOCATABLE :: isTr(:), seen(:)
  REAL(KIND=dp) :: dGL, dTr, frac, nGL, nGLc, nTr, nTrc, dminGL, dminTr, tnow, gm, ct
  INTEGER :: t, i, k, n, nn, minSteps, nCold, nTemp
  INTEGER, SAVE :: nCalls = 0
  LOGICAL :: Found, trigger
  CHARACTER(*), PARAMETER :: Caller = 'RemeshTrigger'
!------------------------------------------------------------------------------
  Params => GetSolverParams()
  Mesh => GetMesh()
  nn = Mesh % NumberOfNodes
  nCalls = nCalls + 1
  dVar => VariableGet( Mesh % Variables, 'Remesh Distance', ThisOnly = .TRUE. )
  gmVar => VariableGet( Mesh % Variables, 'GroundedMask' )
  ctVar => VariableGet( Mesh % Variables, 'ColdTempMask' )
  IF ( .NOT. ( ASSOCIATED(dVar) .AND. ASSOCIATED(gmVar) ) ) &
      CALL Fatal( Caller, 'needs Remesh Distance and GroundedMask' )
  dGL = ListGetConstReal( Params, 'GL Safety Distance', Found )
  IF ( .NOT. Found ) dGL = 5000.0_dp
  dTr = ListGetConstReal( Params, 'Transition Safety Distance', Found )
  IF ( .NOT. Found ) dTr = 1000.0_dp
  frac = ListGetConstReal( Params, 'Transition Fraction', Found )
  IF ( .NOT. Found ) frac = 0.05_dp
  minSteps = ListGetInteger( Params, 'Minimum Steps', Found )
  IF ( .NOT. Found ) minSteps = 1

  ALLOCATE( isTr(nn), seen(nn) )
  isTr = .FALSE.; seen = .FALSE.
  ! the transition: base elements with both a cold and a temperate grounded node
  IF ( ASSOCIATED( ctVar ) ) THEN
    DO t = 1, Solver % NumberOfActiveElements
      Element => GetActiveElement(t)
      n = GetElementNOFNodes( Element )
      nCold = 0; nTemp = 0
      DO k = 1, n
        i = Element % NodeIndexes(k)
        IF ( pval( gmVar, i ) <= 0 .OR. pval( ctVar, i ) <= 0 ) CYCLE
        IF ( nval( gmVar, i ) < -0.5_dp ) CYCLE
        IF ( nval( ctVar, i ) > 0.0_dp ) THEN; nTemp = nTemp + 1; ELSE; nCold = nCold + 1; END IF
      END DO
      IF ( nCold > 0 .AND. nTemp > 0 ) THEN
        DO k = 1, n
          i = Element % NodeIndexes(k)
          IF ( pval( gmVar, i ) > 0 .AND. pval( ctVar, i ) > 0 ) THEN
            IF ( nval( gmVar, i ) > -0.5_dp ) isTr(i) = .TRUE.
          END IF
        END DO
      END IF
    END DO
  END IF
  nGL = 0; nGLc = 0; nTr = 0; nTrc = 0; dminGL = HUGE(1.0_dp); dminTr = HUGE(1.0_dp)
  DO t = 1, Solver % NumberOfActiveElements
    Element => GetActiveElement(t)
    n = GetElementNOFNodes( Element )
    DO k = 1, n
      i = Element % NodeIndexes(k)
      IF ( seen(i) ) CYCLE
      seen(i) = .TRUE.
      IF ( .NOT. own(i) ) CYCLE                  ! shared nodes counted once (by their owner)
      IF ( pval( dVar, i ) <= 0 .OR. pval( gmVar, i ) <= 0 ) CYCLE
      gm = nval( gmVar, i )
      IF ( ABS( gm ) < 0.5_dp ) THEN
        nGL = nGL + 1
        dminGL = MIN( dminGL, nval( dVar, i ) )
        IF ( nval( dVar, i ) < dGL ) nGLc = nGLc + 1
      END IF
      IF ( isTr(i) ) THEN
        nTr = nTr + 1
        dminTr = MIN( dminTr, nval( dVar, i ) )
        IF ( nval( dVar, i ) < dTr ) nTrc = nTrc + 1
      END IF
    END DO
  END DO
  IF ( ParEnv % PEs > 1 ) THEN
    nGL = ParallelReduction( nGL ); nGLc = ParallelReduction( nGLc )
    nTr = ParallelReduction( nTr ); nTrc = ParallelReduction( nTrc )
    dminGL = ParallelReduction( dminGL, 1 ); dminTr = ParallelReduction( dminTr, 1 )
  END IF
  trigger = ( nGLc > 0.5_dp ) .OR. ( nTr > 0.5_dp .AND. nTrc > frac * nTr )
  tnow = 0.0_dp
  tVar => VariableGet( Mesh % Variables, 'Time' )
  IF ( ASSOCIATED( tVar ) ) tnow = tVar % Values(1)
  WRITE( Message, '(A,F10.3,A,I0,A,I0,A,ES9.2,A,I0,A,I0,A,ES9.2,A)' ) 'remesh check t', tnow, ': grounding line ', &
      NINT(nGLc), ' of ', NINT(nGL), ' nodes within the safety distance (closest to coarse mesh ', dminGL, &
      ' m), transition ', NINT(nTrc), ' of ', NINT(nTr), ' (closest ', dminTr, ' m)'
  CALL Info( Caller, Message, Level=3 )
  IF ( trigger .AND. nCalls >= minSteps ) THEN
    WRITE( Message, '(A,F10.3,A,I0)' ) 'REMESH NEEDED at t', tnow, ', step ', nCalls
    CALL Info( Caller, Message, Level=1 )
    CALL ListAddConstReal( Model % Simulation, 'Exit Condition', 1.0_dp )
    IF ( ParEnv % MyPe == 0 ) THEN
      OPEN( 31, FILE='remesh_needed', STATUS='REPLACE' )
      WRITE( 31, '(ES24.16,1X,I0,4(1X,I0))' ) tnow, nCalls, NINT(nGLc), NINT(nGL), NINT(nTrc), NINT(nTr)
      CLOSE( 31 )
    END IF
  END IF
  DEALLOCATE( isTr, seen )

CONTAINS

  FUNCTION own( m ) RESULT( o )
    INTEGER :: m
    LOGICAL :: o
    o = .TRUE.
    IF ( ParEnv % PEs > 1 ) o = ( Mesh % ParallelInfo % NeighbourList(m) % Neighbours(1) == ParEnv % MyPe )
  END FUNCTION own

  FUNCTION pval( v, m ) RESULT( q )
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
!------------------------------------------------------------------------------
END SUBROUTINE RemeshTrigger
!------------------------------------------------------------------------------
