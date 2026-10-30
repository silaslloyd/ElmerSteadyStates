!------------------------------------------------------------------------------
! Shared between FSSSAShelf (fills it) and FSSSAAfterLinsolve (applies it): the passive columns' plug flow, as
! (index in the Flow Solution vector, value) pairs.
!------------------------------------------------------------------------------
MODULE FSSSAShared
  USE Types, ONLY : dp
  IMPLICIT NONE
  INTEGER, ALLOCATABLE :: plugIdx(:)
  REAL(KIND=dp), ALLOCATABLE :: plugVal(:)
  INTEGER :: nplug = 0
END MODULE FSSSAShared


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
  REAL(KIND=dp) :: d2, dmin, m, dmax, dy, pw, strip
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

  ! "Periodic Width" (m): periodic sides, y and y + width the same column; the distance wraps across the seam, so a
  ! node and its periodic image get the same distance and the same active/passive state (without it they could
  ! differ under a wavy grounding line, and the periodic constraint then tied an active node to a passive one)
  pw = ListGetConstReal( Params, 'Periodic Width', Found )
  IF ( .NOT. Found ) pw = 0.0_dp
  strip = ListGetConstReal( Params, 'Periodic Seam Strip', Found )
  IF ( .NOT. Found ) strip = 0.0_dp
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
      dy = ABS( Mesh % Nodes % y(i) - ally(k) )
      IF ( pw > 0.0_dp ) dy = MIN( dy, ABS( pw - dy ) )      ! periodic in y: the distance across the seam
      d2 = (Mesh % Nodes % x(i) - allx(k))**2 + dy**2
      IF ( d2 < dmin ) dmin = d2
    END DO
    dVal(q) = SQRT( dmin )
    ! the seam strip: 0 (always full Stokes) within "Periodic Seam Strip" of the periodic seam. Elmer decides whether a
    ! node is passive-only from its own side's elements, so a seam node and its periodic image could differ (one
    ! fixed at its old value, the other solved) and the periodic constraint did not hold there: one row of elements
    ! on each side of the seam active makes every seam node active on both sides
    IF ( pw > 0.0_dp .AND. strip > 0.0_dp ) THEN
      IF ( MIN( Mesh % Nodes % y(i), pw - Mesh % Nodes % y(i) ) <= strip ) dVal(q) = 0.0_dp
    END IF
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
! at depth d below the surface of the interface column of thickness H (both from the node coordinates): the normal
! stress of a laterally uniform shelf in free flotation, whose depth
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
  USE ParallelUtils, ONLY : ParallelReduction
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t), TARGET :: Solver
  TYPE(Matrix_t), POINTER :: A
  INTEGER :: n, DOFs
  REAL(KIND=dp) :: b(n), x(n), Norm

  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(Element_t), POINTER :: El
  TYPE(Variable_t), POINTER :: dVar
  TYPE(ParallelInfo_t), POINTER :: parInfo
  TYPE(NormalTangential_t), POINTER :: NT
  INTEGER, POINTER :: Perm(:)
  INTEGER, ALLOCATABLE :: tagP(:), lst(:)
  INTEGER :: nn, t, j, k, f, nf, q, c, r, nd, code, nfaces, pass, ni, i, ia, ib
  INTEGER :: fn(4,6)
  REAL(KIND=dp) :: Dcut, rhoi, rhow, g, pts(3,4), av(3), cen(3), fc(3), sig, dep, hei, Fxyz(3), Fr(3), ftot, fsum, dg(8)
  LOGICAL :: Found, passive, inface
  LOGICAL, ALLOCATABLE :: onIF(:)
  REAL(KIND=dp), ALLOCATABLE :: zTop(:), zBot(:)
  LOGICAL, SAVE :: Reported = .FALSE.
  LOGICAL, ALLOCATABLE, SAVE :: prevP(:)       ! passive state of each active element at the previous solve
  INTEGER, SAVE :: ncall = 0
  REAL(KIND=dp) :: npas, nchg, nchgLoc, ngm, nreb
  ! periodic sides: Elmer has already folded each slave node's row into its master rows and replaced it by the
  ! periodic constraint (SetPeriodicBoundariesPass1/2), so a slave node's force goes to its masters, by the
  ! projector weights (pStart/pCnt/pNode/pW: per slave node, its masters)
  INTEGER, ALLOCATABLE, SAVE :: pStart(:), pCnt(:), pNode(:)
  REAL(KIND=dp), ALLOCATABLE, SAVE :: pW(:)
  LOGICAL, SAVE :: pDone = .FALSE.
  TYPE(Matrix_t), POINTER :: Proj
  INTEGER :: bc, ii, l, np, mnode, kk
  REAL(KIND=dp) :: nmiss, wsl
  INTEGER :: thrEl, thrGm
  LOGICAL :: rebuild
  TYPE(Variable_t), POINTER :: gmVar
  INTEGER, ALLOCATABLE, SAVE :: prevGM(:)      ! grounded (1) / floating (-1) at the previous solve
  LOGICAL, SAVE :: gmFirst = .TRUE.
  ! the SSA shelf's stress on the interface (fs_ssa_shelf = "ssa"): its strain rate at the column's base node
  TYPE(Variable_t), POINTER :: eVar, zbVar
  LOGICAL :: ssaStress
  INTEGER :: nlev, nbl, Lb, ibase, qe, nssa
  REAL(KIND=dp) :: mu, exx, eyy, exy, fssa, funi
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
  IF ( .NOT. ASSOCIATED(dVar) ) CALL Fatal( 'FSSSALinsolve', 'needs the variable GL Distance' )
  rhoi = ListGetConstReal( Model % Constants, 'Ice Density' )
  rhow = ListGetConstReal( Model % Constants, 'Water Density' )
  g = ListGetConstReal( Model % Constants, 'Gravity Norm' )
  ssaStress = ListGetLogical( Solver % Values, 'FS SSA Shelf Stress', Found )
  Lb = 0; nbl = 1; mu = 0.0_dp
  IF ( ssaStress ) THEN
    mu = ListGetConstReal( Solver % Values, 'FS SSA Viscosity', UnFoundFatal=.TRUE. )
    eVar => VariableGet( Mesh % Variables, 'SSA Strain' )
    zbVar => VariableGet( Mesh % Variables, 'Zb' )
    nlev = ListGetInteger( Model % Simulation, 'Extruded Mesh Levels', Found )
    IF ( .NOT. ( ASSOCIATED(eVar) .AND. ASSOCIATED(zbVar) ) .OR. nlev < 2 ) THEN
      ssaStress = .FALSE.
    ELSE
      nbl = Mesh % NumberOfNodes / nlev
      DO i = 1, Mesh % NumberOfNodes
        IF ( i > SIZE( zbVar % Perm ) ) EXIT
        IF ( zbVar % Perm(i) > 0 ) THEN
          Lb = (i-1) / nbl + 1
          EXIT
        END IF
      END DO
    END IF
  END IF
  fssa = 0.0_dp; funi = 0.0_dp; nssa = 0

  ! nodes of passive elements (Elmer's rule with Passive Element Min Nodes = 1), exchanged between ranks
  nn = Mesh % NumberOfNodes
  ALLOCATE( tagP(nn) )
  tagP = 0
  IF ( .NOT. pDone ) THEN
    pDone = .TRUE.
    ALLOCATE( pStart(nn), pCnt(nn) )
    pStart = 0; pCnt = 0
    np = 0
    DO pass = 1, 2
      DO bc = 1, Model % NumberOfBCs
        IF ( .NOT. ListGetLogical( Model % BCs(bc) % Values, 'Periodic BC Velocity 1', Found ) ) CYCLE
        Proj => Model % BCs(bc) % PMatrix
        IF ( .NOT. ASSOCIATED( Proj ) ) CYCLE
        DO i = 1, Proj % NumberOfRows
          ii = Proj % InvPerm(i)
          IF ( ii <= 0 .OR. ii > nn ) CYCLE
          DO l = Proj % Rows(i), Proj % Rows(i+1) - 1
            IF ( Proj % Cols(l) <= 0 .OR. Proj % Values(l) == 0.0_dp ) CYCLE
            IF ( pass == 1 ) THEN
              pCnt(ii) = pCnt(ii) + 1
            ELSE
              np = np + 1
              pNode( pStart(ii) + np - 1 ) = Proj % Cols(l)
              pW( pStart(ii) + np - 1 ) = Proj % Values(l)
            END IF
          END DO
          IF ( pass == 2 ) np = 0
        END DO
      END DO
      IF ( pass == 1 ) THEN
        k = 1
        DO i = 1, nn
          pStart(i) = k
          k = k + pCnt(i)
        END DO
        ALLOCATE( pNode( MAX(k-1,1) ), pW( MAX(k-1,1) ) )
      END IF
    END DO
    WRITE( Message, '(A,I0,A)' ) 'periodic slave nodes on this partition: ', COUNT( pCnt > 0 ), &
        ' (interface forces there go to their masters)'
    CALL Info( 'FSSSALinsolve', Message, Level=5 )
  END IF
  nmiss = 0.0_dp
  DO t = 1, Solver % NumberOfActiveElements
    El => Mesh % Elements( Solver % ActiveElements(t) )
    passive = .FALSE.
    DO j = 1, El % TYPE % NumberOfNodes
      IF ( essNodeVal( dVar, El % NodeIndexes(j) ) > Dcut ) passive = .TRUE.
    END DO
    IF ( passive ) tagP( El % NodeIndexes(1:El % TYPE % NumberOfNodes) ) = 1
  END DO
  ! diagnostics: elements that changed between active and passive since the previous solve (a moving interface)
  IF ( .NOT. ALLOCATED( prevP ) ) THEN
    ALLOCATE( prevP( Solver % NumberOfActiveElements ) )
    prevP = .FALSE.
  END IF
  npas = 0.0_dp; nchg = 0.0_dp
  DO t = 1, MIN( Solver % NumberOfActiveElements, SIZE(prevP) )
    El => Mesh % Elements( Solver % ActiveElements(t) )
    passive = .FALSE.
    DO j = 1, El % TYPE % NumberOfNodes
      IF ( essNodeVal( dVar, El % NodeIndexes(j) ) > Dcut ) passive = .TRUE.
    END DO
    IF ( passive ) npas = npas + 1.0_dp
    IF ( ncall > 0 .AND. ( passive .NEQV. prevP(t) ) ) nchg = nchg + 1.0_dp
    prevP(t) = passive
  END DO
  nchgLoc = nchg
  ! the grounded mask's changes too (contact grounding line: a node that grounds or floats changes its rows a lot,
  ! sliding against free flotation): counted like the active-set changes, ILU rebuilt where any happened
  ngm = 0.0_dp
  gmVar => VariableGet( Mesh % Variables, 'GroundedMask' )
  IF ( ASSOCIATED( gmVar ) ) THEN
    IF ( .NOT. ALLOCATED( prevGM ) ) THEN
      ALLOCATE( prevGM(nn) )
      prevGM = 0
      gmFirst = .TRUE.
    END IF
    DO i = 1, MIN( nn, SIZE(prevGM) )
      k = 0
      IF ( ASSOCIATED( gmVar % Perm ) ) THEN
        IF ( i <= SIZE( gmVar % Perm ) ) k = gmVar % Perm(i)
      ELSE
        k = i
      END IF
      IF ( k <= 0 ) CYCLE
      j = MERGE( 1, -1, gmVar % Values(k) >= -0.5_dp )
      IF ( .NOT. gmFirst .AND. j /= prevGM(i) ) ngm = ngm + 1.0_dp
      prevGM(i) = j
    END DO
    gmFirst = .FALSE.
  END IF
  ! rebuild on this rank only above the thresholds ("FS SSA Rebuild Min Elements", "FS SSA Rebuild Min Grounding
  ! Changes", default 1 each: any change): every rank waits for the slowest, so one rank's rebuild costs all of
  ! them its time; a few changed columns cost fewer extra iterations than an ILU4 factorisation
  thrEl = ListGetInteger( Solver % Values, 'FS SSA Rebuild Min Elements', Found )
  IF ( .NOT. Found ) thrEl = 1
  thrGm = ListGetInteger( Solver % Values, 'FS SSA Rebuild Min Grounding Changes', Found )
  IF ( .NOT. Found ) thrGm = 1
  rebuild = ( nchgLoc > 0.0_dp .AND. nchgLoc >= REAL(thrEl, dp) ) .OR. ( ngm > 0.0_dp .AND. ngm >= REAL(thrGm, dp) )
  nreb = MERGE( 1.0_dp, 0.0_dp, rebuild )
  IF ( ParEnv % PEs > 1 ) THEN
    npas = ParallelReduction( npas ); nchg = ParallelReduction( nchg ); ngm = ParallelReduction( ngm )
    nreb = ParallelReduction( nreb )
  END IF
  ! the active set changed on this rank: its ILU factors (kept over "Linear System Precondition Recompute"
  ! solves) were built for the old one, with the then-passive rows fixed; rebuild them for this solve (the
  ! partition-local ILU needs no other rank). Elmer refactorises when the matrix's solve count is a multiple of
  ! the recompute interval: in parallel the counter of the partition's inside matrix, which carries the factors.
  IF ( rebuild ) THEN
    IF ( ASSOCIATED( A % ParMatrix ) ) THEN
      IF ( ASSOCIATED( A % ParMatrix % SplittedMatrix ) ) THEN
        IF ( ASSOCIATED( A % ParMatrix % SplittedMatrix % InsideMatrix ) ) &
            A % ParMatrix % SplittedMatrix % InsideMatrix % SolveCount = 0
      END IF
    ELSE
      A % SolveCount = 0
    END IF
  END IF
  ncall = ncall + 1
  WRITE( Message, '(A,I0,A,I0,A,I0,A,I0,A,I0)' ) 'solve ', ncall, ': passive elements ', NINT(npas), &
      ', changed active/passive since the previous solve ', NINT(nchg), ', grounded/floating ', NINT(ngm), &
      '; ILU factors rebuilt on ranks: ', NINT(nreb)
  CALL Info( 'FSSSALinsolve', Message, Level=3 )
  IF ( ParEnv % PEs > 1 ) THEN
    parInfo => Mesh % ParallelInfo
    CALL CommunicateParallelSystemTag( parInfo, Itag=tagP, ParOper=2 )
  END IF

  ! pass 1: the interface nodes; then the top and base of each interface column (from the node coordinates: the
  ! "height" variable is ambiguous in the template, the Zs solver exports one too); pass 2: the loads
  ALLOCATE( onIF(nn), zTop(nn), zBot(nn) )
  onIF = .FALSE.; zTop = 0.0_dp; zBot = 0.0_dp
  DO pass = 1, 2
  nfaces = 0; ftot = 0.0_dp
  ! diagnostics: sum of face x-areas, area-weighted H, expected force, min/max H, x range of face centres
  dg = 0.0_dp; dg(4) = HUGE(1.0_dp); dg(5) = -HUGE(1.0_dp); dg(6) = HUGE(1.0_dp); dg(7) = -HUGE(1.0_dp)
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
      IF ( pass == 1 ) THEN
        onIF( El % NodeIndexes( fn(:,f) ) ) = .TRUE.
        CYCLE
      END IF
      fc = SUM( pts, DIM=2 ) / 4.0_dp
      IF ( DOT_PRODUCT( av, fc - cen ) < 0.0_dp ) av = -av          ! out of the active element
      nfaces = nfaces + 1
      dg(1) = dg(1) + av(1)
      dg(6) = MIN( dg(6), fc(1) ); dg(7) = MAX( dg(7), fc(1) )
      DO j = 1, 4
        nd = El % NodeIndexes( fn(j,f) )
        IF ( nd > SIZE(Perm) ) CYCLE
        q = Perm(nd)
        IF ( q <= 0 ) CYCLE
        hei = zTop(nd) - zBot(nd)                     ! ice thickness of the interface column
        dep = zTop(nd) - Mesh % Nodes % z(nd)         ! depth below its surface
        sig = -rhoi * g * dep + 0.5_dp * rhoi * g * hei * (1.0_dp - rhoi / rhow)
        dg(2) = dg(2) + hei * av(1) / 4.0_dp
        dg(3) = dg(3) - 0.5_dp * rhoi**2 / rhow * g * hei * av(1) / 4.0_dp
        dg(4) = MIN( dg(4), hei ); dg(5) = MAX( dg(5), hei )
        dg(8) = dg(8) + 1.0_dp
        Fxyz = sig * av / 4.0_dp
        funi = funi + Fxyz(1)
        IF ( ssaStress ) THEN
          ! the SSA shelf's stress: deviatoric from its strain rate (depth-uniform, sigma_xz = sigma_yz = 0) plus
          ! the cryostatic -rho_i g d, on the face's area vector
          ibase = (Lb-1)*nbl + MOD( nd-1, nbl ) + 1
          qe = 0
          IF ( Lb > 0 .AND. ibase >= 1 .AND. ibase <= SIZE( eVar % Perm ) ) qe = eVar % Perm(ibase)
          IF ( qe > 0 ) THEN
            IF ( eVar % Values( eVar % DOFs*(qe-1)+4 ) > 0.5_dp ) THEN
              exx = eVar % Values( eVar % DOFs*(qe-1)+1 )
              eyy = eVar % Values( eVar % DOFs*(qe-1)+2 )
              exy = eVar % Values( eVar % DOFs*(qe-1)+3 )
              Fxyz(1) = ( 2.0_dp*mu*( (2.0_dp*exx + eyy)*av(1) + exy*av(2) ) - rhoi*g*dep*av(1) ) / 4.0_dp
              Fxyz(2) = ( 2.0_dp*mu*( exy*av(1) + (exx + 2.0_dp*eyy)*av(2) ) - rhoi*g*dep*av(2) ) / 4.0_dp
              Fxyz(3) = -rhoi*g*dep*av(3) / 4.0_dp
              nssa = nssa + 1
            END IF
          END IF
        END IF
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
        IF ( pCnt(nd) > 0 ) THEN
          ! a periodic slave: to its masters (rotated to the master's frame)
          DO l = pStart(nd), pStart(nd) + pCnt(nd) - 1
            mnode = pNode(l); wsl = pW(l)
            IF ( mnode > SIZE(Perm) ) THEN
              nmiss = nmiss + 1.0_dp
              CYCLE
            END IF
            IF ( Perm(mnode) <= 0 ) THEN
              nmiss = nmiss + 1.0_dp
              CYCLE
            END IF
            Fr = Fxyz
            kk = 0
            IF ( NT % NormalTangentialNOFNodes > 0 .AND. ASSOCIATED( NT % BoundaryReorder ) ) THEN
              IF ( mnode <= SIZE( NT % BoundaryReorder ) ) kk = NT % BoundaryReorder(mnode)
            END IF
            IF ( kk > 0 ) THEN
              Fr(1) = DOT_PRODUCT( NT % BoundaryNormals(kk,:), Fxyz )
              Fr(2) = DOT_PRODUCT( NT % BoundaryTangent1(kk,:), Fxyz )
              Fr(3) = DOT_PRODUCT( NT % BoundaryTangent2(kk,:), Fxyz )
            END IF
            DO c = 1, 3
              r = DOFs * (Perm(mnode)-1) + c
              IF ( ALLOCATED( A % ConstrainedDOF ) ) THEN
                IF ( A % ConstrainedDOF(r) ) CYCLE
              END IF
              b(r) = b(r) + wsl * Fr(c)
            END DO
          END DO
        ELSE
          DO c = 1, 3
            r = DOFs * (q-1) + c
            IF ( ALLOCATED( A % ConstrainedDOF ) ) THEN
              IF ( A % ConstrainedDOF(r) ) CYCLE
            END IF
            b(r) = b(r) + Fr(c)
          END DO
        END IF
        ftot = ftot + Fxyz(1)
      END DO
    END DO
  END DO
  IF ( pass == 1 ) THEN
    ! top and base of each interface column: the extreme z of the interface nodes at the same (x,y) (whole columns are
    ! on one rank: the mesh is extruded after partitioning)
    ni = COUNT( onIF )
    ALLOCATE( lst(ni) )
    ni = 0
    DO i = 1, nn
      IF ( onIF(i) ) THEN
        ni = ni + 1; lst(ni) = i
      END IF
    END DO
    DO ia = 1, ni
      i = lst(ia)
      zTop(i) = Mesh % Nodes % z(i); zBot(i) = zTop(i)
      DO ib = 1, ni
        k = lst(ib)
        IF ( ABS( Mesh % Nodes % x(k) - Mesh % Nodes % x(i) ) < 1.0e-3_dp .AND. &
             ABS( Mesh % Nodes % y(k) - Mesh % Nodes % y(i) ) < 1.0e-3_dp ) THEN
          zTop(i) = MAX( zTop(i), Mesh % Nodes % z(k) ); zBot(i) = MIN( zBot(i), Mesh % Nodes % z(k) )
        END IF
      END DO
    END DO
    DEALLOCATE( lst )
  END IF
  END DO
  IF ( ParEnv % PEs > 1 ) nmiss = ParallelReduction( nmiss )
  IF ( nmiss > 0.0_dp ) THEN
    WRITE( Message, '(A,I0,A)' ) 'WARNING: ', NINT(nmiss), ' interface force contributions at periodic slave nodes '// &
        'whose master is on another partition were dropped'
    CALL Info( 'FSSSALinsolve', Message, Level=3 )
  END IF
  IF ( ssaStress ) THEN
    fssa = ftot
    npas = REAL( nssa, dp )
    IF ( ParEnv % PEs > 1 ) THEN
      fssa = ParallelReduction( fssa ); funi = ParallelReduction( funi ); npas = ParallelReduction( npas )
    END IF
    WRITE( Message, '(A,ES12.4,A,ES12.4,A,I0,A)' ) 'interface force x: SSA shelf ', fssa, &
        '  (free-floating shelf formula ', funi, '; ', NINT(npas), ' node contributions from the SSA)'
    CALL Info( 'FSSSALinsolve', Message, Level=3 )
  END IF
  IF ( .NOT. Reported ) THEN
    ! totals over all ranks (the interface usually lies on a few ranks only)
    fsum = REAL( nfaces, dp )
    IF ( ParEnv % PEs > 1 ) THEN
      fsum = ParallelReduction( fsum )
      ftot = ParallelReduction( ftot )
      DO j = 1, 3
        dg(j) = ParallelReduction( dg(j) )
      END DO
      dg(8) = ParallelReduction( dg(8) )
      dg(4) = ParallelReduction( dg(4), 1 ); dg(6) = ParallelReduction( dg(6), 1 )
      dg(5) = ParallelReduction( dg(5), 2 ); dg(7) = ParallelReduction( dg(7), 2 )
    END IF
    WRITE( Message, '(A,I0,A,ES12.4,A,ES12.4)' ) 'shelf force on ', NINT(fsum), ' interface faces (all ranks), x component ', &
        ftot, '  (expected -rho_w g D^2/2 integrated: ', dg(3)
    CALL Info( 'FSSSALinsolve', Message, Level=3 )
    WRITE( Message, '(A,ES12.4,A,ES12.4,A,ES12.4,A,ES12.4,A,ES12.4,A,ES12.4)' ) 'interface x-area ', dg(1), &
        '  area-weighted H ', dg(2) / MAX( dg(1), 1.0_dp ), '  H min ', dg(4), '  max ', dg(5), '  at x ', dg(6), ' to ', dg(7)
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


!------------------------------------------------------------------------------
! FSSSAShelf: the shelf beyond ESS's moving full-Stokes region ([elmer] fs_ssa_km with fs_ssa_shelf = "ssa") by the
! shallow-shelf approximation, every time step before Stokes. Active on the base of the ice (the "interface" body:
! its boundary elements are the solver's elements), 2 DOFs ("SSA Velocity").
!
! The SSA domain: the base faces of the passive ice (a face with a node more than "FS SSA Distance" from the
! grounding line, as the passive Stokes elements). Its nodes that also belong to a face of the full-Stokes region
! (the interface) take the depth-averaged full-Stokes velocity of their column (Dirichlet; the last Stokes solution,
! so lagged by a time step); every other base node outside the domain too (unused). With walls ("FS SSA Walls",
! free-slip sides) v = 0 on the side nodes. Linear viscous ice (constant viscosity mu, "FS SSA Viscosity", as the
! templates' Newtonian ice), floating, no friction:
!     div( mu H [ 2 (2 u_x + v_y), u_y + v_x ; u_y + v_x, 2 (u_x + 2 v_y) ] ) = rho_i g H grad s,
! with the floating ice's driving stress written as grad P, P = rho_i g (1 - rho_i/rho_w) H^2 / 2, and integrated by
! parts: the calving front's sea-water condition is then natural (no front term), on any front shape.
!
! Before the solve the passive columns are put back in flotation (base -rho_i/rho_w H, surface (1 - rho_i/rho_w) H,
! thickness kept: nothing else holds them there, Stokes being passive). After it, the passive columns' nodes get
! the SSA velocity as plug flow in "Flow Solution", which Elmer keeps fixed on passive nodes: the free surfaces
! advect the passive shelf with it, and ice that becomes active again starts from it. The vertical velocity:
! w = u . grad(z_b) - (z - z_b) div(u) (incompressible; the thickness then changes as dH/dt = a - div(u H)), the
! pressure cryostatic plus the SSA's vertical normal stress, p = rho_i g (z_s - z) - 2 mu div(u).
! "SSA Strain" (exported, 4 DOFs on the base nodes: e_xx, e_yy, e_xy and 1 where defined) is the nodal average of
! the SSA strain rate over the domain's faces: FSSSALinsolve's interface stress ("FS SSA Shelf Stress").
!
! Columns: Elmer's internal extrusion numbers the nodes layer by layer (node i in column mod(i-1, nodes per layer)+1);
! the ice of a column runs from the layer of the Zb nodes to that of the Zs nodes; whole columns are on one rank.
! Solver section:
!   Procedure = File "GLDistance" "FSSSAShelf"     Variable = -dofs 2 "SSA Velocity"
!   Exported Variable 1 = -dofs 4 "SSA Strain"
!   FS SSA Distance = Real (m)    FS SSA Viscosity = Real    FS SSA Walls = Logical (default True)
!   Linear System Solver = Direct ...
!------------------------------------------------------------------------------
SUBROUTINE FSSSAShelf( Model, Solver, dt, Transient )
  USE DefUtils
  USE FSSSAShared
  USE MeshBasics, ONLY : CommunicateParallelSystemTag
  USE ParallelUtils, ONLY : ParallelReduction, ParallelSumNodalVector
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t) :: Solver
  REAL(KIND=dp) :: dt
  LOGICAL :: Transient

  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(Element_t), POINTER :: El
  TYPE(ValueList_t), POINTER :: Params
  TYPE(Variable_t), POINTER :: dVar, zbVar, zsVar, fVar, gmVar, eVar
  TYPE(Matrix_t), POINTER :: A
  TYPE(ParallelInfo_t), POINTER :: parInfo
  TYPE(Nodes_t), SAVE :: Nodes
  TYPE(GaussIntegrationPoints_t) :: IP
  INTEGER, POINTER :: sPerm(:), fPerm(:)
  INTEGER, POINTER, SAVE :: idPerm(:) => NULL()
  INTEGER :: nn, nlev, nbl, Lb, Lt, i, j, k, t, n, p, q, lc, L, ib, it, nd, ninner, ndir, nmax, fd
  INTEGER, ALLOCATABLE :: tagS(:), tagA(:)
  LOGICAL, ALLOCATABLE :: inner(:), elS(:)
  REAL(KIND=dp), ALLOCATABLE :: Hc(:), ubar(:), vbar(:), zbc(:), STIFF(:,:), FORCE(:), Basis(:), dBasisdx(:,:)
  REAL(KIND=dp), ALLOCATABLE :: nodal(:,:), wsum(:)
  REAL(KIND=dp) :: Dcut, mu, rhoi, rhow, g, r, detJ, Hg, Pg, ymin, ymax, tolw, wq, uq(2), dux(2), duy(2), &
      dzb(2), ar, zb, zs, H, divu, wb, su, sv, umax, Norm, relax
  LOGICAL :: Found, walls, stat, ssaEl
  CHARACTER(*), PARAMETER :: Caller = 'FSSSAShelf'
!------------------------------------------------------------------------------
  Mesh => Solver % Mesh
  Params => GetSolverParams()
  sPerm => Solver % Variable % Perm
  nn = Mesh % NumberOfNodes
  fd = Solver % Variable % DOFs
  IF ( fd /= 2 ) CALL Fatal( Caller, 'the variable needs 2 DOFs (SSA Velocity 1, 2)' )

  Dcut = ListGetConstReal( Params, 'FS SSA Distance', UnFoundFatal=.TRUE. )
  mu = ListGetConstReal( Params, 'FS SSA Viscosity', UnFoundFatal=.TRUE. )
  walls = ListGetLogical( Params, 'FS SSA Walls', Found )
  IF ( .NOT. Found ) walls = .TRUE.
  rhoi = ListGetConstReal( Model % Constants, 'Ice Density' )
  rhow = ListGetConstReal( Model % Constants, 'Water Density' )
  g = ListGetConstReal( Model % Constants, 'Gravity Norm' )
  r = rhoi / rhow

  dVar => VariableGet( Mesh % Variables, 'GL Distance' )
  zbVar => VariableGet( Mesh % Variables, 'Zb' )
  zsVar => VariableGet( Mesh % Variables, 'Zs' )
  fVar => VariableGet( Mesh % Variables, 'Flow Solution' )
  gmVar => VariableGet( Mesh % Variables, 'GroundedMask' )
  eVar => VariableGet( Mesh % Variables, 'SSA Strain' )
  IF ( .NOT. ( ASSOCIATED(dVar) .AND. ASSOCIATED(zbVar) .AND. ASSOCIATED(zsVar) .AND. ASSOCIATED(fVar) ) ) &
      CALL Fatal( Caller, 'needs the variables GL Distance, Zb, Zs and Flow Solution' )
  fPerm => fVar % Perm

  nlev = ListGetInteger( Model % Simulation, 'Extruded Mesh Levels', Found )
  IF ( .NOT. Found .OR. nlev < 2 .OR. MOD(nn, MAX(nlev,1)) /= 0 ) CALL Fatal( Caller, 'Extruded Mesh Levels unknown' )
  nbl = nn / nlev
  ! the layers of the ice base and surface
  Lb = 0; Lt = 0
  DO i = 1, nn
    IF ( Lb == 0 .AND. essPerm( zbVar, i ) > 0 ) Lb = (i-1) / nbl + 1
    IF ( Lt == 0 .AND. essPerm( zsVar, i ) > 0 ) Lt = (i-1) / nbl + 1
    IF ( Lb > 0 .AND. Lt > 0 ) EXIT
  END DO
  IF ( Lb == 0 .OR. Lt <= Lb ) CALL Fatal( Caller, 'no ice columns (Zb / Zs layers) on this partition' )

  IF ( .NOT. ASSOCIATED( idPerm ) ) THEN
    ALLOCATE( idPerm(nn) )
    DO i = 1, nn
      idPerm(i) = i
    END DO
  END IF

  ! the SSA faces (a node beyond the distance) and the nodes of full-Stokes faces, exchanged between ranks
  ALLOCATE( tagS(nn), tagA(nn), elS(Solver % NumberOfActiveElements) )
  tagS = 0; tagA = 0
  DO t = 1, Solver % NumberOfActiveElements
    El => Mesh % Elements( Solver % ActiveElements(t) )
    n = El % TYPE % NumberOfNodes
    ssaEl = .FALSE.
    DO j = 1, n
      IF ( essVal( dVar, El % NodeIndexes(j) ) > Dcut ) ssaEl = .TRUE.
    END DO
    elS(t) = ssaEl
    IF ( ssaEl ) THEN
      tagS( El % NodeIndexes(1:n) ) = 1
    ELSE
      tagA( El % NodeIndexes(1:n) ) = 1
    END IF
  END DO
  IF ( ParEnv % PEs > 1 ) THEN
    parInfo => Mesh % ParallelInfo
    CALL CommunicateParallelSystemTag( parInfo, Itag=tagS, ParOper=2 )
    CALL CommunicateParallelSystemTag( parInfo, Itag=tagA, ParOper=2 )
  END IF

  ! per column: interior of the SSA domain; thickness (passive columns floated first); depth-averaged Stokes velocity
  ALLOCATE( inner(nbl), Hc(nbl), ubar(nbl), vbar(nbl), zbc(nbl) )
  inner = .FALSE.; Hc = 0.0_dp; ubar = 0.0_dp; vbar = 0.0_dp; zbc = 0.0_dp
  ninner = 0
  DO lc = 1, nbl
    ib = (Lb-1)*nbl + lc
    it = (Lt-1)*nbl + lc
    IF ( essPerm( zbVar, ib ) <= 0 .OR. essPerm( zsVar, it ) <= 0 ) CYCLE
    zb = zbVar % Values( essPerm( zbVar, ib ) )
    zs = zsVar % Values( essPerm( zsVar, it ) )
    H = zs - zb
    inner(lc) = ( tagS(ib) > 0 .AND. tagA(ib) == 0 )
    IF ( inner(lc) .AND. ASSOCIATED( gmVar ) ) THEN
      IF ( essVal( gmVar, ib ) >= -0.5_dp ) inner(lc) = .FALSE.     ! grounded: never (distance 0)
    END IF
    IF ( inner(lc) ) THEN
      zb = -r * H
      zbVar % Values( essPerm( zbVar, ib ) ) = zb
      zsVar % Values( essPerm( zsVar, it ) ) = (1.0_dp - r) * H
      ninner = ninner + 1
    END IF
    Hc(lc) = H
    zbc(lc) = zb
    ! depth average of the full-Stokes velocity (layers equally spaced: the trapezoidal rule)
    su = 0.0_dp; sv = 0.0_dp
    DO L = Lb, Lt
      i = (L-1)*nbl + lc
      q = 0
      IF ( i <= SIZE(fPerm) ) q = fPerm(i)
      IF ( q <= 0 ) CYCLE
      wq = 1.0_dp
      IF ( L == Lb .OR. L == Lt ) wq = 0.5_dp
      su = su + wq * fVar % Values( fVar % DOFs*(q-1) + 1 )
      sv = sv + wq * fVar % Values( fVar % DOFs*(q-1) + 2 )
    END DO
    ubar(lc) = su / REAL( Lt - Lb, dp )
    vbar(lc) = sv / REAL( Lt - Lb, dp )
  END DO

  ! the sides (walls: v = 0 there)
  ymin = MINVAL( Mesh % Nodes % y(1:nn) ); ymax = MAXVAL( Mesh % Nodes % y(1:nn) )
  IF ( ParEnv % PEs > 1 ) THEN
    ymin = ParallelReduction( ymin, 1 ); ymax = ParallelReduction( ymax, 2 )
  END IF
  tolw = 1.0e-6_dp * MAX( ymax - ymin, 1.0_dp )

  ! assembly over the SSA faces (the face projected onto the horizontal plane)
  nmax = Model % MaxElementNodes
  ALLOCATE( STIFF(2*nmax,2*nmax), FORCE(2*nmax), Basis(nmax), dBasisdx(nmax,3) )
  IF ( .NOT. ASSOCIATED( Nodes % x ) ) ALLOCATE( Nodes % x(nmax), Nodes % y(nmax), Nodes % z(nmax) )
  CALL DefaultInitialize()
  DO t = 1, Solver % NumberOfActiveElements
    IF ( .NOT. elS(t) ) CYCLE
    El => GetActiveElement(t)
    n = GetElementNOFNodes()
    Nodes % x(1:n) = Mesh % Nodes % x( El % NodeIndexes(1:n) )
    Nodes % y(1:n) = Mesh % Nodes % y( El % NodeIndexes(1:n) )
    Nodes % z(1:n) = 0.0_dp
    STIFF = 0.0_dp; FORCE = 0.0_dp
    IP = GaussPoints( El )
    DO k = 1, IP % n
      stat = ElementInfo( El, Nodes, IP % U(k), IP % V(k), IP % W(k), detJ, Basis, dBasisdx )
      Hg = 0.0_dp
      DO p = 1, n
        Hg = Hg + Basis(p) * Hc( MOD( El % NodeIndexes(p)-1, nbl ) + 1 )
      END DO
      Hg = MAX( Hg, 1.0_dp )
      Pg = 0.5_dp * rhoi * g * (1.0_dp - r) * Hg**2
      wq = IP % s(k) * detJ
      DO p = 1, n
        DO q = 1, n
          STIFF(2*p-1,2*q-1) = STIFF(2*p-1,2*q-1) + wq * mu * Hg * &
              ( 4.0_dp*dBasisdx(p,1)*dBasisdx(q,1) + dBasisdx(p,2)*dBasisdx(q,2) )
          STIFF(2*p-1,2*q) = STIFF(2*p-1,2*q) + wq * mu * Hg * &
              ( 2.0_dp*dBasisdx(p,1)*dBasisdx(q,2) + dBasisdx(p,2)*dBasisdx(q,1) )
          STIFF(2*p,2*q-1) = STIFF(2*p,2*q-1) + wq * mu * Hg * &
              ( dBasisdx(p,1)*dBasisdx(q,2) + 2.0_dp*dBasisdx(p,2)*dBasisdx(q,1) )
          STIFF(2*p,2*q) = STIFF(2*p,2*q) + wq * mu * Hg * &
              ( dBasisdx(p,1)*dBasisdx(q,1) + 4.0_dp*dBasisdx(p,2)*dBasisdx(q,2) )
        END DO
        FORCE(2*p-1) = FORCE(2*p-1) + wq * Pg * dBasisdx(p,1)
        FORCE(2*p) = FORCE(2*p) + wq * Pg * dBasisdx(p,2)
      END DO
    END DO
    CALL DefaultUpdateEquations( STIFF(1:2*n,1:2*n), FORCE(1:2*n) )
  END DO
  CALL DefaultFinishBulkAssembly()
  CALL DefaultFinishAssembly()

  ! Dirichlet: the depth-averaged Stokes velocity outside the SSA interior, v = 0 on the walls
  A => Solver % Matrix
  ndir = 0
  DO i = 1, MIN( nn, SIZE(sPerm) )
    q = sPerm(i)
    IF ( q <= 0 ) CYCLE
    lc = MOD( i-1, nbl ) + 1
    IF ( .NOT. inner(lc) ) THEN
      CALL UpdateDirichletDof( A, 2*(q-1)+1, ubar(lc) )
      CALL UpdateDirichletDof( A, 2*(q-1)+2, vbar(lc) )
      ndir = ndir + 1
    ELSE IF ( walls .AND. ( ABS( Mesh % Nodes % y(i) - ymin ) < tolw .OR. ABS( Mesh % Nodes % y(i) - ymax ) < tolw ) ) THEN
      CALL UpdateDirichletDof( A, 2*(q-1)+2, 0.0_dp )
    END IF
  END DO
  CALL DefaultDirichletBCs()
  Norm = DefaultSolve()

  ! nodal averages over the SSA faces: the strain rate, div(u) and grad(z_b) (area weighted), summed over ranks
  ALLOCATE( nodal(nn,7) )
  nodal = 0.0_dp
  DO t = 1, Solver % NumberOfActiveElements
    IF ( .NOT. elS(t) ) CYCLE
    El => Mesh % Elements( Solver % ActiveElements(t) )
    n = El % TYPE % NumberOfNodes
    Nodes % x(1:n) = Mesh % Nodes % x( El % NodeIndexes(1:n) )
    Nodes % y(1:n) = Mesh % Nodes % y( El % NodeIndexes(1:n) )
    Nodes % z(1:n) = 0.0_dp
    IP = GaussPoints( El )
    ar = 0.0_dp; dux = 0.0_dp; duy = 0.0_dp; dzb = 0.0_dp
    DO k = 1, IP % n
      stat = ElementInfo( El, Nodes, IP % U(k), IP % V(k), IP % W(k), detJ, Basis, dBasisdx )
      wq = IP % s(k) * detJ
      ar = ar + wq
      DO p = 1, n
        nd = El % NodeIndexes(p)
        q = sPerm(nd)
        IF ( q <= 0 ) CYCLE
        uq(1) = Solver % Variable % Values( 2*(q-1)+1 )
        uq(2) = Solver % Variable % Values( 2*(q-1)+2 )
        dux = dux + wq * uq * dBasisdx(p,1)        ! (u_x, v_x)
        duy = duy + wq * uq * dBasisdx(p,2)        ! (u_y, v_y)
        dzb(1) = dzb(1) + wq * zbc( MOD(nd-1, nbl) + 1 ) * dBasisdx(p,1)
        dzb(2) = dzb(2) + wq * zbc( MOD(nd-1, nbl) + 1 ) * dBasisdx(p,2)
      END DO
    END DO
    IF ( ar <= 0.0_dp ) CYCLE
    DO p = 1, n
      nd = El % NodeIndexes(p)
      nodal(nd,1) = nodal(nd,1) + dux(1)                       ! e_xx * area
      nodal(nd,2) = nodal(nd,2) + duy(2)                       ! e_yy * area
      nodal(nd,3) = nodal(nd,3) + 0.5_dp * ( duy(1) + dux(2) ) ! e_xy * area
      nodal(nd,4) = nodal(nd,4) + dux(1) + duy(2)              ! div u * area
      nodal(nd,5) = nodal(nd,5) + dzb(1)
      nodal(nd,6) = nodal(nd,6) + dzb(2)
      nodal(nd,7) = nodal(nd,7) + ar
    END DO
  END DO
  IF ( ParEnv % PEs > 1 ) THEN
    DO k = 1, 7
      CALL ParallelSumNodalVector( Mesh, nodal(:,k), idPerm, Solver % Matrix )
    END DO
  END IF
  DO i = 1, nn
    IF ( nodal(i,7) > 0.0_dp ) nodal(i,1:6) = nodal(i,1:6) / nodal(i,7)
  END DO
  ! the strain rate for the interface stress, optionally relaxed in time ("FS SSA Stress Relaxation" w, default 1:
  ! none; e = e_previous + w (e_SSA - e_previous) where the node had one). The coupling is explicit (the SSA's inflow
  ! is the last Stokes velocity, Stokes' interface stress the SSA's strain rate); it was stable unrelaxed once the
  ! periodic seam's forces went to the master nodes (the blow-up with periodic sides was that, not the coupling)
  relax = ListGetConstReal( Params, 'FS SSA Stress Relaxation', Found )
  IF ( .NOT. Found ) relax = 1.0_dp
  IF ( ASSOCIATED( eVar ) ) THEN
    DO i = 1, MIN( nn, SIZE(eVar % Perm) )
      q = eVar % Perm(i)
      IF ( q <= 0 ) CYCLE
      j = eVar % DOFs*(q-1)
      IF ( eVar % DOFs >= 4 .AND. eVar % Values(j+4) > 0.5_dp .AND. nodal(i,7) > 0.0_dp ) THEN
        eVar % Values(j+1:j+3) = eVar % Values(j+1:j+3) + relax * ( nodal(i,1:3) - eVar % Values(j+1:j+3) )
      ELSE
        eVar % Values(j+1:j+3) = nodal(i,1:3)
      END IF
      IF ( eVar % DOFs >= 4 ) eVar % Values(j+4) = MERGE( 1.0_dp, 0.0_dp, nodal(i,7) > 0.0_dp )
    END DO
  END IF

  ! the passive columns: plug flow from the SSA solution in Flow Solution (Elmer keeps it on passive nodes, but in
  ! the normal-tangential frame on the base nodes: FSSSAAfterLinsolve writes it back after every Stokes solve)
  umax = 0.0_dp
  IF ( ALLOCATED( plugIdx ) ) DEALLOCATE( plugIdx, plugVal )
  ALLOCATE( plugIdx( MAX( 1, COUNT(inner) * (Lt-Lb+1) * fVar % DOFs ) ) )
  ALLOCATE( plugVal( SIZE(plugIdx) ) )
  nplug = 0
  DO lc = 1, nbl
    IF ( .NOT. inner(lc) ) CYCLE
    ib = (Lb-1)*nbl + lc
    it = (Lt-1)*nbl + lc
    q = sPerm(ib)
    IF ( q <= 0 ) CYCLE
    uq(1) = Solver % Variable % Values( 2*(q-1)+1 )
    uq(2) = Solver % Variable % Values( 2*(q-1)+2 )
    umax = MAX( umax, SQRT( SUM( uq**2 ) ) )
    divu = nodal(ib,4)
    wb = uq(1) * nodal(ib,5) + uq(2) * nodal(ib,6)
    DO L = Lb, Lt
      i = (L-1)*nbl + lc
      k = 0
      IF ( i <= SIZE(fPerm) ) k = fPerm(i)
      IF ( k <= 0 ) CYCLE
      j = fVar % DOFs * (k-1)
      fVar % Values( j+1 ) = uq(1)
      fVar % Values( j+2 ) = uq(2)
      fVar % Values( j+3 ) = wb - ( Mesh % Nodes % z(i) - Mesh % Nodes % z(ib) ) * divu
      IF ( fVar % DOFs >= 4 ) fVar % Values( j+4 ) = rhoi * g * ( Mesh % Nodes % z(it) - Mesh % Nodes % z(i) ) &
          - 2.0_dp * mu * divu
      DO p = 1, fVar % DOFs
        nplug = nplug + 1
        plugIdx(nplug) = j + p
        plugVal(nplug) = fVar % Values( j+p )
      END DO
    END DO
  END DO
  IF ( ParEnv % PEs > 1 ) THEN
    su = ParallelReduction( REAL( ninner, dp ) )
    umax = ParallelReduction( umax, 2 )
  ELSE
    su = REAL( ninner, dp )
  END IF
  WRITE( Message, '(A,I0,A,ES11.4,A)' ) 'shelf columns by SSA (floated, plug flow): ', NINT(su), &
      ', largest SSA speed ', umax, ' m/a'
  CALL Info( Caller, Message, Level=3 )

CONTAINS

  FUNCTION essPerm( v, i ) RESULT( p )
    TYPE(Variable_t), POINTER :: v
    INTEGER :: i, p
    p = i
    IF ( ASSOCIATED( v % Perm ) ) THEN
      p = 0
      IF ( i <= SIZE( v % Perm ) ) p = v % Perm(i)
    END IF
  END FUNCTION essPerm

  FUNCTION essVal( v, i ) RESULT( val )
    TYPE(Variable_t), POINTER :: v
    INTEGER :: i
    REAL(KIND=dp) :: val
    INTEGER :: p
    val = 0.0_dp
    p = essPerm( v, i )
    IF ( p <= 0 .OR. p > SIZE( v % Values ) ) RETURN
    val = v % Values(p)
  END FUNCTION essVal
!------------------------------------------------------------------------------
END SUBROUTINE FSSSAShelf
!------------------------------------------------------------------------------


!------------------------------------------------------------------------------
! FSSSAAfterLinsolve: "After Linsolve" of the Stokes solver with the SSA shelf (fs_ssa_shelf = "ssa"): the passive
! columns get back FSSSAShelf's plug flow. Elmer fixes passive-only nodes at their values but, on the base, applies
! them in the normal-tangential frame and rotates back, which would scramble them; those nodes are decoupled from
! the active ones, so this changes nothing else. Runs after Elmer's back-rotation (Cartesian x).
!------------------------------------------------------------------------------
INTEGER FUNCTION FSSSAAfterLinsolve( Model, Solver, A, b, x, n, DOFs, Norm )
  USE DefUtils
  USE FSSSAShared
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t) :: Solver
  TYPE(Matrix_t), POINTER :: A
  INTEGER :: n, DOFs
  REAL(KIND=dp) :: b(n), x(n), Norm
  INTEGER :: k
!------------------------------------------------------------------------------
  FSSSAAfterLinsolve = 0
  DO k = 1, nplug
    IF ( plugIdx(k) >= 1 .AND. plugIdx(k) <= n ) x( plugIdx(k) ) = plugVal(k)
  END DO
!------------------------------------------------------------------------------
END FUNCTION FSSSAAfterLinsolve
!------------------------------------------------------------------------------
