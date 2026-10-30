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


!------------------------------------------------------------------------------
! FSSSALinsolve: "Before Linsolve" of the Stokes solver with ESS's moving full-Stokes region ([elmer] fs_ssa_km).
! Adds to the right-hand side the force of the free-floating shelf beyond the region on its interface, instead of
! Elmer's "Passive Target" boundary (GetPassiveBoundary reallocates the mesh's elements and leaves 3D edge pointers
! dangling: the Stokes solve diverged at once). The interface: the vertical faces of active Stokes elements whose
! nodes all also belong to passive elements (an element is passive when any of its nodes is more than
! "FS SSA Distance" from the grounding line: "GL Distance" > FS SSA Distance, as Passive Element Min Nodes = 1).
! The traction is sigma_nn n (n out of the active region), sigma_nn = -rho_i g d + rho_i g H (1 - rho_i/rho_w)/2
! at depth d (H = d + height): the normal stress of a laterally uniform shelf in free flotation, whose depth
! integral balances the sea water on the shelf front (exact for an unbuttressed shelf). Nodal forces: face area/4 per
! node (quadrilateral faces of the extruded hexahedra or wedges), rotated to the normal-tangential frame where the
! system is, and not on constrained (Dirichlet) rows. Passive-element membership of nodes on partition boundaries is
! exchanged between ranks. The rows get the force before scaling (Before Linsolve runs before it).
! Also rotates the initial guess to the normal-tangential frame ("FS SSA Rotate Guess", as RotateGuess.F90), since
! a solver has only one Before Linsolve procedure.
! Solver (Stokes) section:
!   Before Linsolve = File "GLDistance" "FSSSALinsolve"
!   FS SSA Distance = Real (m)          FS SSA Rotate Guess = Logical
!------------------------------------------------------------------------------
INTEGER FUNCTION FSSSALinsolve( Model, Solver, A, b, x, n, DOFs, Norm )
  USE DefUtils
  USE MeshBasics, ONLY : CommunicateParallelSystemTag
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t), TARGET :: Solver
  TYPE(Matrix_t), POINTER :: A
  INTEGER :: n, DOFs
  REAL(KIND=dp) :: b(n), x(n), Norm

  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(Element_t), POINTER :: El
  TYPE(Variable_t), POINTER :: dVar, depVar, heiVar
  TYPE(ParallelInfo_t), POINTER :: parInfo
  TYPE(NormalTangential_t), POINTER :: NT
  INTEGER, POINTER :: Perm(:)
  INTEGER, ALLOCATABLE :: tagP(:)
  INTEGER :: nn, t, j, k, f, nf, q, c, r, nd, code, nfaces
  INTEGER :: fn(4,6)
  REAL(KIND=dp) :: Dcut, rhoi, rhow, g, pts(3,4), av(3), cen(3), fc(3), sig, dep, hei, Fxyz(3), Fr(3), ftot
  LOGICAL :: Found, passive, inface
  LOGICAL, SAVE :: Reported = .FALSE.
!------------------------------------------------------------------------------
  FSSSALinsolve = 0
  Mesh => Solver % Mesh
  Perm => Solver % Variable % Perm
  NT => Solver % NormalTangential

  IF ( ListGetLogical( Solver % Values, 'FS SSA Rotate Guess', Found ) ) THEN
    IF ( NT % NormalTangentialNOFNodes > 0 ) CALL RotateNTSystemAll( x, Perm, DOFs )
  END IF

  Dcut = ListGetConstReal( Solver % Values, 'FS SSA Distance', Found )
  IF ( .NOT. Found ) RETURN
  dVar => VariableGet( Mesh % Variables, 'GL Distance' )
  depVar => VariableGet( Mesh % Variables, 'depth' )
  heiVar => VariableGet( Mesh % Variables, 'height' )
  IF ( .NOT. ( ASSOCIATED(dVar) .AND. ASSOCIATED(depVar) .AND. ASSOCIATED(heiVar) ) ) &
      CALL Fatal( 'FSSSALinsolve', 'needs the variables GL Distance, depth and height' )
  rhoi = ListGetConstReal( Model % Constants, 'Ice Density' )
  rhow = ListGetConstReal( Model % Constants, 'Water Density' )
  g = ListGetConstReal( Model % Constants, 'Gravity Norm' )

  ! nodes of passive elements (Elmer's rule with Passive Element Min Nodes = 1), exchanged between ranks
  nn = Mesh % NumberOfNodes
  ALLOCATE( tagP(nn) )
  tagP = 0
  DO t = 1, Solver % NumberOfActiveElements
    El => Mesh % Elements( Solver % ActiveElements(t) )
    passive = .FALSE.
    DO j = 1, El % TYPE % NumberOfNodes
      IF ( essNodeVal( dVar, El % NodeIndexes(j) ) > Dcut ) passive = .TRUE.
    END DO
    IF ( passive ) tagP( El % NodeIndexes(1:El % TYPE % NumberOfNodes) ) = 1
  END DO
  IF ( ParEnv % PEs > 1 ) THEN
    parInfo => Mesh % ParallelInfo
    CALL CommunicateParallelSystemTag( parInfo, Itag=tagP, ParOper=2 )
  END IF

  nfaces = 0; ftot = 0.0_dp
  DO t = 1, Solver % NumberOfActiveElements
    El => Mesh % Elements( Solver % ActiveElements(t) )
    code = El % TYPE % ElementCode
    passive = .FALSE.
    DO j = 1, El % TYPE % NumberOfNodes
      IF ( essNodeVal( dVar, El % NodeIndexes(j) ) > Dcut ) passive = .TRUE.
    END DO
    IF ( passive ) CYCLE
    SELECT CASE( code / 100 )
    CASE( 8 )      ! hexahedron: its six faces
      nf = 6
      fn(:,1) = (/1,2,3,4/); fn(:,2) = (/5,6,7,8/); fn(:,3) = (/1,2,6,5/)
      fn(:,4) = (/2,3,7,6/); fn(:,5) = (/3,4,8,7/); fn(:,6) = (/4,1,5,8/)
    CASE( 7 )      ! wedge: its three quadrilateral (vertical) faces
      nf = 3
      fn(:,1) = (/1,2,5,4/); fn(:,2) = (/2,3,6,5/); fn(:,3) = (/3,1,4,6/)
    CASE DEFAULT
      CYCLE
    END SELECT
    cen(1) = SUM( Mesh % Nodes % x( El % NodeIndexes(1:El % TYPE % NumberOfNodes) ) ) / El % TYPE % NumberOfNodes
    cen(2) = SUM( Mesh % Nodes % y( El % NodeIndexes(1:El % TYPE % NumberOfNodes) ) ) / El % TYPE % NumberOfNodes
    cen(3) = SUM( Mesh % Nodes % z( El % NodeIndexes(1:El % TYPE % NumberOfNodes) ) ) / El % TYPE % NumberOfNodes
    DO f = 1, nf
      inface = .TRUE.
      DO j = 1, 4
        IF ( tagP( El % NodeIndexes( fn(j,f) ) ) == 0 ) inface = .FALSE.
      END DO
      IF ( .NOT. inface ) CYCLE
      DO j = 1, 4
        nd = El % NodeIndexes( fn(j,f) )
        pts(1,j) = Mesh % Nodes % x(nd); pts(2,j) = Mesh % Nodes % y(nd); pts(3,j) = Mesh % Nodes % z(nd)
      END DO
      ! area vector of the quadrilateral: half the cross product of its diagonals
      av = 0.5_dp * essCross( pts(:,3) - pts(:,1), pts(:,4) - pts(:,2) )
      IF ( ABS(av(3)) > 0.5_dp * SQRT( SUM(av**2) ) ) CYCLE        ! a horizontal face (top/base): not the interface
      fc = SUM( pts, DIM=2 ) / 4.0_dp
      IF ( DOT_PRODUCT( av, fc - cen ) < 0.0_dp ) av = -av          ! out of the active element
      nfaces = nfaces + 1
      DO j = 1, 4
        nd = El % NodeIndexes( fn(j,f) )
        IF ( nd > SIZE(Perm) ) CYCLE
        q = Perm(nd)
        IF ( q <= 0 ) CYCLE
        dep = essNodeVal( depVar, nd ); hei = essNodeVal( heiVar, nd )
        sig = -rhoi * g * dep + 0.5_dp * rhoi * g * (dep + hei) * (1.0_dp - rhoi / rhow)
        Fxyz = sig * av / 4.0_dp
        Fr = Fxyz
        k = 0
        IF ( NT % NormalTangentialNOFNodes > 0 .AND. ASSOCIATED( NT % BoundaryReorder ) ) THEN
          IF ( nd <= SIZE( NT % BoundaryReorder ) ) k = NT % BoundaryReorder(nd)
        END IF
        IF ( k > 0 ) THEN
          Fr(1) = DOT_PRODUCT( NT % BoundaryNormals(k,:), Fxyz )
          Fr(2) = DOT_PRODUCT( NT % BoundaryTangent1(k,:), Fxyz )
          Fr(3) = DOT_PRODUCT( NT % BoundaryTangent2(k,:), Fxyz )
        END IF
        DO c = 1, 3
          r = DOFs * (q-1) + c
          IF ( ALLOCATED( A % ConstrainedDOF ) ) THEN
            IF ( A % ConstrainedDOF(r) ) CYCLE
          END IF
          b(r) = b(r) + Fr(c)
        END DO
        ftot = ftot + Fxyz(1)
      END DO
    END DO
  END DO
  IF ( .NOT. Reported ) THEN
    WRITE( Message, '(A,I0,A,ES12.4)' ) 'shelf force on ', nfaces, ' interface faces (this rank), x component ', ftot
    CALL Info( 'FSSSALinsolve', Message, Level=3 )
    Reported = .TRUE.
  END IF

CONTAINS

  FUNCTION essNodeVal( v, i ) RESULT( val )
    TYPE(Variable_t), POINTER :: v
    INTEGER :: i
    REAL(KIND=dp) :: val
    INTEGER :: p
    val = 0.0_dp
    p = i
    IF ( ASSOCIATED( v % Perm ) ) THEN
      IF ( i > SIZE( v % Perm ) ) RETURN
      p = v % Perm(i)
    END IF
    IF ( p <= 0 .OR. p > SIZE( v % Values ) ) RETURN
    val = v % Values(p)
  END FUNCTION essNodeVal

  FUNCTION essCross( a1, a2 ) RESULT( cr )
    REAL(KIND=dp) :: a1(3), a2(3), cr(3)
    cr(1) = a1(2)*a2(3) - a1(3)*a2(2)
    cr(2) = a1(3)*a2(1) - a1(1)*a2(3)
    cr(3) = a1(1)*a2(2) - a1(2)*a2(1)
  END FUNCTION essCross
!------------------------------------------------------------------------------
END FUNCTION FSSSALinsolve
!------------------------------------------------------------------------------
