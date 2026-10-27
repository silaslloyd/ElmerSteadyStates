!------------------------------------------------------------------------------
!> PeriodicUtils: make nodal fields periodic across implicit periodic BCs.
!>
!> The nodes linked by the periodic projectors (Model % BCs(k) % PMatrix: row = image node, column = master
!> node, Elmer SetPeriodicBoundariesPass1) are grouped into chains (a corner node can be the image in two
!> periodic BCs). PeriodicSymmetrize then gives every member of a group the same value, by Mode:
!>   "folded"  nodal loads from a solver's matrix ("Calculate Loads"): Elmer adds each image row into its
!>             master's row, so the root (the member that is no image) holds the group total and the images
!>             their own one-sided loads. Every member gets  sum over the non-image members / members.
!>   "split"   nodal integrals that were not folded (each node its own share, e.g. GetHydrostaticLoads):
!>             every member gets  sum over all members / members.
!>   "mean"    = "split" (averages, e.g. normal vectors; renormalise afterwards if they must be unit vectors)
!>   "fold"    what Elmer's folding does to a non-folded field: the root gets the sum over all members, the
!>             images keep their own values (only to emulate the old ElmerStreams halve/double treatment)
!>   "min", "max"   e.g. a grounded mask (min: a periodic node is floating / at the grounding line if any
!>             of its copies is)
!> Parallel: both nodes of a pair must be on the same partition (ElmerGrid -periodic). Groups without a
!> local root are left as they are (counted in the return value norphan).
!------------------------------------------------------------------------------
MODULE PeriodicUtils
  USE DefUtils
  IMPLICIT NONE

CONTAINS

  !> root of node k in the union-find forest
  FUNCTION PGRoot( parent, k ) RESULT( r )
    INTEGER, INTENT(IN) :: parent(:)
    INTEGER, INTENT(IN) :: k
    INTEGER :: r
    r = k
    DO WHILE ( parent(r) /= r )
      r = parent(r)
    END DO
  END FUNCTION PGRoot

  !> the periodic groups of the mesh nodes: parent (union-find; root(i) = PGRoot(parent, i)), isImage (the
  !> node is an image in some periodic BC), inPair (the node belongs to some pair); npairs projector rows
  !> used, nweak rows without a single master of weight ~1 (non-conforming periodic mesh)
  SUBROUTINE PeriodicGroups( Model, nn, parent, isImage, inPair, npairs, nweak )
    TYPE(Model_t) :: Model
    INTEGER, INTENT(IN) :: nn
    INTEGER, ALLOCATABLE :: parent(:)
    LOGICAL, ALLOCATABLE :: isImage(:), inPair(:)
    INTEGER, INTENT(OUT) :: npairs, nweak

    TYPE(Matrix_t), POINTER :: P
    INTEGER :: bc, i, l, img, mst, a, b
    REAL(KIND=dp) :: wmax

    IF ( ALLOCATED( parent ) ) DEALLOCATE( parent )
    IF ( ALLOCATED( isImage ) ) DEALLOCATE( isImage )
    IF ( ALLOCATED( inPair ) ) DEALLOCATE( inPair )
    ALLOCATE( parent(nn), isImage(nn), inPair(nn) )
    DO i = 1, nn
      parent(i) = i
    END DO
    isImage = .FALSE.
    inPair = .FALSE.
    npairs = 0
    nweak = 0
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
        a = PGRoot( parent, img )
        b = PGRoot( parent, mst )
        IF ( a /= b ) parent(a) = b
      END DO
    END DO
  END SUBROUTINE PeriodicGroups

  !> make Values (dofs per node, through Perm) equal on all members of each periodic group, by Mode (above).
  !> Returns the number of periodic pairs found (0: nothing done) and, optionally, the nodes left unchanged
  !> because their group has no root on this partition ("folded" / "fold" only).
  SUBROUTINE PeriodicSymmetrize( Model, Values, Perm, dofs, Mode, npairs, norphan )
    TYPE(Model_t) :: Model
    REAL(KIND=dp) :: Values(:)
    INTEGER :: Perm(:)
    INTEGER, INTENT(IN) :: dofs
    CHARACTER(LEN=*), INTENT(IN) :: Mode
    INTEGER, INTENT(OUT) :: npairs
    INTEGER, OPTIONAL, INTENT(OUT) :: norphan

    INTEGER, ALLOCATABLE :: parent(:), nmem(:), nroot(:)
    LOGICAL, ALLOCATABLE :: isImage(:), inPair(:), seen(:)
    REAL(KIND=dp), ALLOCATABLE :: acc(:,:)
    INTEGER :: nn, i, a, d, p, nweak, norph, k
    CHARACTER(*), PARAMETER :: Caller = 'PeriodicSymmetrize'
    LOGICAL :: UseRoots

    nn = Model % Mesh % NumberOfNodes
    CALL PeriodicGroups( Model, nn, parent, isImage, inPair, npairs, nweak )
    IF ( PRESENT( norphan ) ) norphan = 0
    IF ( npairs == 0 ) RETURN
    IF ( nweak > 0 ) CALL Warn( Caller, I2S(nweak)//' projector rows without a single master of weight 1 '// &
        '(non-conforming periodic mesh?): the strongest master was used' )

    SELECT CASE ( TRIM(Mode) )
    CASE ( 'folded', 'split', 'mean', 'fold', 'min', 'max' )
    CASE DEFAULT
      CALL Fatal( Caller, 'Unknown mode "'//TRIM(Mode)//'": folded, split, mean, fold, min or max' )
    END SELECT
    UseRoots = ( TRIM(Mode) == 'folded' )

    ALLOCATE( acc(dofs, nn), nmem(nn), nroot(nn), seen(nn) )
    nmem = 0
    nroot = 0
    seen = .FALSE.
    SELECT CASE ( TRIM(Mode) )
    CASE ( 'min' )
      acc = HUGE( 1.0_dp )
    CASE ( 'max' )
      acc = -HUGE( 1.0_dp )
    CASE DEFAULT
      acc = 0.0_dp
    END SELECT

    ! ---- accumulate over each group
    DO i = 1, nn
      IF ( .NOT. inPair(i) ) CYCLE
      IF ( i > SIZE( Perm ) ) CYCLE
      p = Perm(i)
      IF ( p <= 0 ) CYCLE
      a = PGRoot( parent, i )
      nmem(a) = nmem(a) + 1
      IF ( .NOT. isImage(i) ) nroot(a) = nroot(a) + 1
      DO d = 1, dofs
        k = dofs*(p-1) + d
        SELECT CASE ( TRIM(Mode) )
        CASE ( 'min' )
          acc(d, a) = MIN( acc(d, a), Values(k) )
        CASE ( 'max' )
          acc(d, a) = MAX( acc(d, a), Values(k) )
        CASE DEFAULT
          IF ( .NOT. UseRoots .OR. .NOT. isImage(i) ) acc(d, a) = acc(d, a) + Values(k)
        END SELECT
      END DO
    END DO

    ! ---- write back
    norph = 0
    DO i = 1, nn
      IF ( .NOT. inPair(i) ) CYCLE
      IF ( i > SIZE( Perm ) ) CYCLE
      p = Perm(i)
      IF ( p <= 0 ) CYCLE
      a = PGRoot( parent, i )
      IF ( nmem(a) == 0 ) CYCLE
      SELECT CASE ( TRIM(Mode) )
      CASE ( 'folded' )
        IF ( nroot(a) == 0 ) THEN
          norph = norph + 1
          CYCLE
        END IF
        DO d = 1, dofs
          Values( dofs*(p-1) + d ) = acc(d, a) / nmem(a)
        END DO
      CASE ( 'split', 'mean' )
        DO d = 1, dofs
          Values( dofs*(p-1) + d ) = acc(d, a) / nmem(a)
        END DO
      CASE ( 'fold' )
        IF ( nroot(a) == 0 ) THEN
          norph = norph + 1
          CYCLE
        END IF
        IF ( .NOT. isImage(i) ) THEN
          DO d = 1, dofs
            Values( dofs*(p-1) + d ) = acc(d, a) / nroot(a)
          END DO
        END IF
      CASE ( 'min', 'max' )
        DO d = 1, dofs
          Values( dofs*(p-1) + d ) = acc(d, a)
        END DO
      END SELECT
    END DO
    IF ( PRESENT( norphan ) ) norphan = norph
    IF ( norph > 0 ) CALL Warn( Caller, I2S(norph)//' nodes whose periodic group has no root on this '// &
        'partition: left unchanged (partition with ElmerGrid -periodic)' )

    DEALLOCATE( parent, isImage, inPair, acc, nmem, nroot, seen )
  END SUBROUTINE PeriodicSymmetrize

END MODULE PeriodicUtils
