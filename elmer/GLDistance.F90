!------------------------------------------------------------------------------
! GLDistance: the horizontal distance of every node to the grounding line, for ESS's moving FS-SSA region
! ([elmer] fs_ssa_km). Run before every time step, active on the ice body: its variable ("GL Distance") is 0 in
! grounded columns and, in floating columns, the distance in (x, y) to the nearest grounding-line node.
!
! The grounding line: the grounded nodes (GroundedMask >= 0) of elements that also have a floating node
! (GroundedMask < 0), from the mask on the ice base. A column's mask is the value at any of its nodes where the
! mask is defined (the mesh comes from Elmer's internal extrusion: nodes numbered layer by layer, so node i is in
! column mod(i-1, nodes per layer)+1; checked). All ranks' grounding-line nodes are gathered on every rank (they
! are few: about two per footprint node across the width), so the distance is exact across partitions. Cost: one
! pass over the elements and one over the floating nodes times the grounding-line nodes.
!
! Solver section:
!   Procedure = File "GLDistance" "GLDistance"
!   Variable = -nooutput "GL Distance"
!   Mask Variable = String "GroundedMask"            (default)
!   Extruded Mesh Levels = Integer                   (default: the Simulation's)
!------------------------------------------------------------------------------
SUBROUTINE GLDistance( Model, Solver, dt, Transient )
  USE DefUtils
  USE SParIterComm      ! MPI (Elmer's own module, built with MPI; elmerf90 has no MPI module path)
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t) :: Solver
  REAL(KIND=dp) :: dt
  LOGICAL :: Transient

  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(Variable_t), POINTER :: gmVar
  TYPE(Element_t), POINTER :: El
  TYPE(ValueList_t), POINTER :: Params
  INTEGER, POINTER :: dPerm(:)
  REAL(KIND=dp), POINTER :: dVal(:)
  INTEGER :: nn, nlev, nbl, i, j, k, t, n, q, lc, ngl, ntot, ierr, nranks, comm
  INTEGER, ALLOCATABLE :: cnts(:), displs(:)
  REAL(KIND=dp), ALLOCATABLE :: colgm(:), glx(:), gly(:), allx(:), ally(:)
  LOGICAL, ALLOCATABLE :: isgl(:)
  LOGICAL :: Found, hasg, hasf
  REAL(KIND=dp) :: d2, dmin, m, dmax
  CHARACTER(:), ALLOCATABLE :: mname
  CHARACTER(*), PARAMETER :: Caller = 'GLDistance'
!------------------------------------------------------------------------------
  Mesh => Solver % Mesh
  Params => GetSolverParams()
  dPerm => Solver % Variable % Perm
  dVal => Solver % Variable % Values
  nn = Mesh % NumberOfNodes

  mname = ListGetString( Params, 'Mask Variable', Found )
  IF ( .NOT. Found ) mname = 'GroundedMask'
  gmVar => VariableGet( Mesh % Variables, mname )
  IF ( .NOT. ASSOCIATED( gmVar ) ) CALL Fatal( Caller, 'no variable '//TRIM(mname) )

  nlev = ListGetInteger( Params, 'Extruded Mesh Levels', Found )
  IF ( .NOT. Found ) nlev = ListGetInteger( Model % Simulation, 'Extruded Mesh Levels', Found )
  IF ( .NOT. Found .OR. nlev < 2 ) CALL Fatal( Caller, 'Extruded Mesh Levels unknown' )
  IF ( MOD(nn, nlev) /= 0 ) CALL Fatal( Caller, 'node count not a multiple of the extruded levels' )
  nbl = nn / nlev

  ! the mask per column (huge: undefined)
  ALLOCATE( colgm(nbl) )
  colgm = HUGE(1.0_dp)
  DO i = 1, nn
    q = 0
    IF ( ASSOCIATED( gmVar % Perm ) ) THEN
      IF ( i <= SIZE( gmVar % Perm ) ) q = gmVar % Perm(i)
    ELSE
      q = i
    END IF
    IF ( q <= 0 ) CYCLE
    colgm( MOD(i-1, nbl) + 1 ) = gmVar % Values(q)
  END DO

  ! the grounding-line nodes: grounded nodes of elements with grounded and floating nodes
  ALLOCATE( isgl(nn) )
  isgl = .FALSE.
  DO t = 1, Mesh % NumberOfBulkElements
    El => Mesh % Elements(t)
    n = El % TYPE % NumberOfNodes
    hasg = .FALSE.; hasf = .FALSE.
    DO j = 1, n
      m = colgm( MOD(El % NodeIndexes(j)-1, nbl) + 1 )
      IF ( m == HUGE(1.0_dp) ) CYCLE
      IF ( m >= -0.5_dp ) hasg = .TRUE.
      IF ( m < -0.5_dp ) hasf = .TRUE.
    END DO
    IF ( .NOT. (hasg .AND. hasf) ) CYCLE
    DO j = 1, n
      k = El % NodeIndexes(j)
      m = colgm( MOD(k-1, nbl) + 1 )
      IF ( m /= HUGE(1.0_dp) .AND. m >= -0.5_dp ) isgl( MOD(k-1, nbl) + 1 ) = .TRUE.   ! by column
    END DO
  END DO
  ngl = COUNT( isgl(1:nbl) )
  ALLOCATE( glx(MAX(ngl,1)), gly(MAX(ngl,1)) )
  k = 0
  DO lc = 1, nbl
    IF ( .NOT. isgl(lc) ) CYCLE
    k = k + 1
    glx(k) = Mesh % Nodes % x(lc)
    gly(k) = Mesh % Nodes % y(lc)
  END DO

  ! gather them on every rank
  IF ( ParEnv % PEs > 1 ) THEN
    comm = ParEnv % ActiveComm
    CALL MPI_Comm_size( comm, nranks, ierr )
    ALLOCATE( cnts(nranks), displs(nranks) )
    CALL MPI_Allgather( ngl, 1, MPI_INTEGER, cnts, 1, MPI_INTEGER, comm, ierr )
    displs(1) = 0
    DO i = 2, nranks
      displs(i) = displs(i-1) + cnts(i-1)
    END DO
    ntot = displs(nranks) + cnts(nranks)
    ALLOCATE( allx(MAX(ntot,1)), ally(MAX(ntot,1)) )
    CALL MPI_Allgatherv( glx, ngl, MPI_DOUBLE_PRECISION, allx, cnts, displs, MPI_DOUBLE_PRECISION, comm, ierr )
    CALL MPI_Allgatherv( gly, ngl, MPI_DOUBLE_PRECISION, ally, cnts, displs, MPI_DOUBLE_PRECISION, comm, ierr )
  ELSE
    ntot = ngl
    ALLOCATE( allx(MAX(ntot,1)), ally(MAX(ntot,1)) )
    allx(1:ntot) = glx(1:ntot); ally(1:ntot) = gly(1:ntot)
  END IF

  ! the distance: 0 in grounded (and undefined) columns, to the nearest grounding-line node in floating ones
  dmax = 0.0_dp
  DO i = 1, nn
    IF ( i > SIZE(dPerm) ) EXIT
    q = dPerm(i)
    IF ( q <= 0 ) CYCLE
    m = colgm( MOD(i-1, nbl) + 1 )
    IF ( m == HUGE(1.0_dp) .OR. m >= -0.5_dp .OR. ntot == 0 ) THEN
      dVal(q) = 0.0_dp
      CYCLE
    END IF
    dmin = HUGE(1.0_dp)
    DO k = 1, ntot
      d2 = (Mesh % Nodes % x(i) - allx(k))**2 + (Mesh % Nodes % y(i) - ally(k))**2
      IF ( d2 < dmin ) dmin = d2
    END DO
    dVal(q) = SQRT( dmin )
    dmax = MAX( dmax, dVal(q) )
  END DO
  IF ( ParEnv % PEs > 1 ) THEN
    CALL MPI_Allreduce( MPI_IN_PLACE, dmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, ParEnv % ActiveComm, ierr )
  END IF
  WRITE( Message, '(A,I0,A,ES10.3,A)' ) 'grounding-line nodes: ', ntot, ', largest distance ', dmax, ' m'
  CALL Info( Caller, Message, Level=3 )
!------------------------------------------------------------------------------
END SUBROUTINE GLDistance
!------------------------------------------------------------------------------
