!------------------------------------------------------------------------------
!> FixPeriodicLoads: a periodic copy of a nodal-loads variable (e.g. "Flow Solution Loads").
!>
!> Why: with an implicit periodic BC, Elmer adds the image node's row into the master node's row of the
!> matrix and RHS, including the saved bulk copy that "Calculate Loads" uses (SetPeriodicBoundariesPass1), and
!> then replaces only the live image row with the constraint. So the computed loads are
!>     master: its own load + the image's load (the total of the periodic node),
!>     image:  its own, one-sided load (it includes the force across the periodic seam).
!> Neither is the load of a node with its own share of the boundary. Here the nodes linked by the periodic
!> projectors (Model % BCs(k) % PMatrix: row = image node, column = master node) are grouped into chains (a
!> corner node can be the image in two periodic BCs); the group total is the sum of the loads of the members
!> that are not images of anything (normally the one root, which holds everything folded into it), and every
!> member gets total / members. Exact for a conforming periodic mesh whose members own equal boundary areas.
!>
!> The source is not changed: the result goes into a new variable (same Perm, same DOFs, components
!> "<target> 1..n"), so solvers that expect the raw loads keep them. Run it after the solver that computes the
!> loads, and before whatever uses them; it can run any number of times.
!>
!> Keywords (Solver section):
!>   Loads Variable  = String "Flow Solution Loads"           (default)
!>   Target Variable = String "Periodic Flow Solution Loads"  (default: "Periodic " // loads variable)
!>
!> Parallel: needs both nodes of a periodic pair on the same partition (ElmerGrid -periodic ...). Groups whose
!> root is not on this partition are left as they are and counted in the log.
!------------------------------------------------------------------------------
SUBROUTINE FixPeriodicLoads( Model, Solver, dt, Transient )
!------------------------------------------------------------------------------
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t) :: Solver
  REAL(KIND=dp) :: dt
  LOGICAL :: Transient
!------------------------------------------------------------------------------
  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(ValueList_t), POINTER :: Params
  TYPE(Variable_t), POINTER :: Src, Tgt
  TYPE(Matrix_t), POINTER :: P
  INTEGER, ALLOCATABLE :: parent(:), nmem(:), nroot(:)
  LOGICAL, ALLOCATABLE :: isImage(:), inPair(:)
  REAL(KIND=dp), ALLOCATABLE :: total(:,:)
  INTEGER :: nn, bc, i, l, img, mst, a, b, d, dofs, ps, ngroups, nweak, norphan, nmulti, npairs
  REAL(KIND=dp) :: wmax
  CHARACTER(LEN=MAX_NAME_LEN) :: sname, tname
  CHARACTER(*), PARAMETER :: Caller = 'FixPeriodicLoads'
  LOGICAL :: Found
!------------------------------------------------------------------------------
  Params => GetSolverParams()
  Mesh => GetMesh()

  sname = ListGetString( Params, 'Loads Variable', Found )
  IF ( .NOT. Found ) sname = 'Flow Solution Loads'
  tname = ListGetString( Params, 'Target Variable', Found )
  IF ( .NOT. Found ) tname = 'Periodic '//TRIM(sname)

  Src => VariableGet( Mesh % Variables, TRIM(sname), ThisOnly = .TRUE. )
  IF ( .NOT. ASSOCIATED( Src ) ) THEN
    CALL Warn( Caller, 'No variable "'//TRIM(sname)//'" (yet): nothing done' )
    RETURN
  END IF
  IF ( .NOT. ASSOCIATED( Src % Perm ) ) CALL Fatal( Caller, '"'//TRIM(sname)//'" has no Perm' )
  dofs = Src % DOFs

  Tgt => VariableGet( Mesh % Variables, TRIM(tname), ThisOnly = .TRUE. )
  IF ( .NOT. ASSOCIATED( Tgt ) ) THEN
    CALL VariableAddVector( Mesh % Variables, Mesh, Solver, TRIM(tname), dofs, Perm = Src % Perm )
    Tgt => VariableGet( Mesh % Variables, TRIM(tname), ThisOnly = .TRUE. )
    CALL Info( Caller, 'Created "'//TRIM(tname)//'" with '//I2S(dofs)//' components', Level=5 )
  END IF
  IF ( SIZE( Tgt % Values ) /= SIZE( Src % Values ) ) &
      CALL Fatal( Caller, '"'//TRIM(tname)//'" and "'//TRIM(sname)//'" differ in size' )
  Tgt % Values = Src % Values

  ! ---- the periodic pairs, joined into groups (union-find over mesh nodes)
  nn = Mesh % NumberOfNodes
  ALLOCATE( parent(nn), isImage(nn), inPair(nn) )
  DO i = 1, nn
    parent(i) = i
  END DO
  isImage = .FALSE.
  inPair = .FALSE.
  nweak = 0
  npairs = 0
  DO bc = 1, Model % NumberOfBCs
    P => Model % BCs(bc) % PMatrix
    IF ( .NOT. ASSOCIATED( P ) ) CYCLE
    IF ( .NOT. ASSOCIATED( P % InvPerm ) ) CYCLE
    DO i = 1, P % NumberOfRows
      img = P % InvPerm(i)
      IF ( img <= 0 .OR. img > nn ) CYCLE
      mst = 0
      wmax = 0.0_dp
      DO l = P % Rows(i), P % Rows(i+1) - 1
        IF ( P % Cols(l) <= 0 .OR. P % Cols(l) > nn ) CYCLE
        IF ( ABS( P % Values(l) ) > wmax ) THEN
          wmax = ABS( P % Values(l) )
          mst = P % Cols(l)
        END IF
      END DO
      IF ( mst == 0 .OR. mst == img ) CYCLE
      IF ( wmax < 0.999_dp ) nweak = nweak + 1
      npairs = npairs + 1
      isImage(img) = .TRUE.
      inPair(img) = .TRUE.
      inPair(mst) = .TRUE.
      a = Root( img )
      b = Root( mst )
      IF ( a /= b ) parent(a) = b
    END DO
  END DO

  IF ( npairs == 0 ) THEN
    CALL Info( Caller, 'No periodic pairs: "'//TRIM(tname)//'" = "'//TRIM(sname)//'"', Level=5 )
    DEALLOCATE( parent, isImage, inPair )
    RETURN
  END IF

  ! ---- group totals: the loads of the members that are not images (where everything was folded to)
  ALLOCATE( total(dofs, nn), nmem(nn), nroot(nn) )
  total = 0.0_dp
  nmem = 0
  nroot = 0
  DO i = 1, nn
    IF ( .NOT. inPair(i) ) CYCLE
    IF ( i > SIZE( Src % Perm ) ) CYCLE
    ps = Src % Perm(i)
    IF ( ps <= 0 ) CYCLE
    a = Root( i )
    nmem(a) = nmem(a) + 1
    IF ( .NOT. isImage(i) ) THEN
      nroot(a) = nroot(a) + 1
      DO d = 1, dofs
        total(d, a) = total(d, a) + Src % Values( dofs*(ps-1) + d )
      END DO
    END IF
  END DO

  ! ---- every member: total / members
  ngroups = 0
  norphan = 0
  nmulti = 0
  DO i = 1, nn
    IF ( .NOT. inPair(i) ) CYCLE
    IF ( i > SIZE( Src % Perm ) ) CYCLE
    ps = Src % Perm(i)
    IF ( ps <= 0 ) CYCLE
    a = Root( i )
    IF ( a == i ) THEN
      ngroups = ngroups + 1
      IF ( nroot(a) > 1 ) nmulti = nmulti + 1
    END IF
    IF ( nroot(a) == 0 ) THEN
      norphan = norphan + 1
      CYCLE
    END IF
    DO d = 1, dofs
      Tgt % Values( dofs*(ps-1) + d ) = total(d, a) / nmem(a)
    END DO
  END DO

  CALL Info( Caller, '"'//TRIM(tname)//'": '//I2S(npairs)//' periodic pairs in '//I2S(ngroups)//' groups', &
      Level=4 )
  IF ( nweak > 0 ) CALL Warn( Caller, I2S(nweak)//' projector rows without a single master of weight 1 '// &
      '(non-conforming periodic mesh?): the strongest master was used' )
  IF ( norphan > 0 ) CALL Warn( Caller, I2S(norphan)//' nodes whose periodic group has no root on this '// &
      'partition: left unchanged (partition with ElmerGrid -periodic)' )
  IF ( nmulti > 0 ) CALL Warn( Caller, I2S(nmulti)//' groups with more than one root: their roots were summed' )

  DEALLOCATE( parent, isImage, inPair, total, nmem, nroot )

CONTAINS

  FUNCTION Root( k ) RESULT( r )
    INTEGER, INTENT(IN) :: k
    INTEGER :: r
    r = k
    DO WHILE ( parent(r) /= r )
      r = parent(r)
    END DO
  END FUNCTION Root

!------------------------------------------------------------------------------
END SUBROUTINE FixPeriodicLoads
!------------------------------------------------------------------------------
