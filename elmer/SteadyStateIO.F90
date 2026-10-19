!------------------------------------------------------------------------------
! SteadyStateIO: exact binary read/write of nodal fields for ElmerSteadyStates.
!
! Two solver procedures:
!
!   StateIO   reads or writes nodal variables, in double precision, keyed by
!             global node number, so the Python driver can set the state
!             (Temperature, Zs, frozen masks, ...) of a run and read it back
!             without any interpolation or ASCII rounding.
!   ESSNoOp   does nothing; used to own exported variables that are only ever
!             set by StateIO (frozen masks, Zb).
!
! Solver keywords (StateIO):
!   State Mode = String "read" | "write"
!   State File = File "name"          ("none" or absent: do nothing)
!   State Variable 1 = String "Temperature"   (write mode; 2, 3, ... as needed)
!   Write Coordinates = Logical True  (write mode: also write coordinate 1..3)
!   Timestep Suffix = Logical True    (write mode: "<State File>_t<timestep>.<partition>",
!                                      for snapshots with Exec Solver = "After Timestep")
!   Strict = Logical True             (read mode: every node of a variable
!                                      that is in the file must be found)
!
! Write mode writes one file per partition, "<State File>.<partition>"
! (also in serial: "<State File>.0"). Read mode reads one merged file (every
! partition reads all of it and picks its own nodes).
!
! File format (stream, native little-endian):
!   int32 magic (1162089265), int32 version (1), int32 nrec
!   nrec times: int32 namelen, char(namelen) name, int64 n,
!               int64 gid(n), float64 value(n)
! In read mode the gids of every record must be sorted ascending.
!
! Build:  elmerf90 -o SteadyStateIO.so SteadyStateIO.F90
!------------------------------------------------------------------------------

SUBROUTINE StateIO( Model, Solver, dt, TransientSimulation )
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t), TARGET :: Solver
  REAL(KIND=dp) :: dt
  LOGICAL :: TransientSimulation

  INTEGER, PARAMETER :: ESS_MAGIC = 1162089265, ESS_VERSION = 1
  TYPE(ValueList_t), POINTER :: Params
  TYPE(Mesh_t), POINTER :: Mesh
  CHARACTER(:), ALLOCATABLE :: fname, mode
  CHARACTER(*), PARAMETER :: Caller = 'StateIO'
  LOGICAL :: Found
  TYPE(Variable_t), POINTER :: TsVar

  Params => GetSolverParams()
  Mesh => Solver % Mesh

  fname = ListGetString( Params, 'State File', Found )
  IF ( .NOT. Found ) RETURN
  IF ( LEN_TRIM(fname) == 0 .OR. TRIM(fname) == 'none' ) RETURN

  mode = ListGetString( Params, 'State Mode', Found )
  IF ( .NOT. Found ) CALL Fatal( Caller, 'Give "State Mode" = "read" or "write"' )

  SELECT CASE ( TRIM(mode) )
  CASE ( 'read' )
    CALL ReadState( TRIM(fname) )
  CASE ( 'write' )
    IF ( ListGetLogical( Params, 'Timestep Suffix', Found ) ) THEN
      TsVar => VariableGet( Mesh % Variables, 'Timestep' )
      IF ( ASSOCIATED( TsVar ) ) fname = TRIM(fname)//'_t'//I2S(NINT(TsVar % Values(1)))
    END IF
    CALL WriteState( TRIM(fname) )
  CASE DEFAULT
    CALL Fatal( Caller, 'Unknown State Mode: '//TRIM(mode) )
  END SELECT

CONTAINS

  FUNCTION NodeGid( i ) RESULT( g )
    INTEGER :: i
    INTEGER(KIND=8) :: g
    g = i
    IF ( ParEnv % PEs > 1 ) THEN
      IF ( ASSOCIATED( Mesh % ParallelInfo % GlobalDOFs ) ) g = Mesh % ParallelInfo % GlobalDOFs(i)
    END IF
  END FUNCTION NodeGid


  FUNCTION FindGid( gids, n, g ) RESULT( k )
    ! binary search in a sorted list; 0 if absent
    INTEGER(KIND=8) :: gids(:), n, g
    INTEGER(KIND=8) :: k, lo, hi, mid
    k = 0
    lo = 1
    hi = n
    DO WHILE ( lo <= hi )
      mid = (lo + hi) / 2
      IF ( gids(mid) == g ) THEN
        k = mid
        RETURN
      ELSE IF ( gids(mid) < g ) THEN
        lo = mid + 1
      ELSE
        hi = mid - 1
      END IF
    END DO
  END FUNCTION FindGid


  SUBROUTINE ReadState( fname )
    CHARACTER(*) :: fname
    INTEGER :: u, ios, magic, version, nrec, r, nlen, i, nset, nmiss, j, l
    INTEGER(KIND=8) :: n, k
    INTEGER(KIND=8), ALLOCATABLE :: gids(:)
    REAL(KIND=dp), ALLOCATABLE :: vals(:)
    CHARACTER(:), ALLOCATABLE :: vname
    TYPE(Variable_t), POINTER :: Var
    LOGICAL :: Strict, Found

    Strict = ListGetLogical( Params, 'Strict', Found )
    IF ( .NOT. Found ) Strict = .TRUE.

    OPEN( NEWUNIT=u, FILE=fname, ACCESS='STREAM', FORM='UNFORMATTED', &
        STATUS='OLD', ACTION='READ', IOSTAT=ios )
    IF ( ios /= 0 ) CALL Fatal( Caller, 'Cannot open state file: '//fname )

    READ( u ) magic, version, nrec
    IF ( magic /= ESS_MAGIC ) CALL Fatal( Caller, 'Not an ESS state file: '//fname )
    IF ( version /= ESS_VERSION ) CALL Fatal( Caller, 'Unsupported state file version' )

    DO r = 1, nrec
      READ( u ) nlen
      IF ( ALLOCATED(vname) ) DEALLOCATE( vname )
      ALLOCATE( CHARACTER(nlen) :: vname )
      READ( u ) vname
      READ( u ) n
      IF ( ALLOCATED(gids) ) DEALLOCATE( gids, vals )
      ALLOCATE( gids(MAX(n,1_8)), vals(MAX(n,1_8)) )
      IF ( n > 0 ) THEN
        READ( u ) gids(1:n)
        READ( u ) vals(1:n)
      END IF

      Var => VariableGet( Mesh % Variables, vname, ThisOnly = .TRUE. )
      IF ( .NOT. ASSOCIATED( Var ) ) THEN
        CALL Fatal( Caller, 'Variable in state file not found in the model: '//vname )
      END IF
      IF ( Var % DOFs /= 1 ) CALL Fatal( Caller, 'Only scalar variables supported: '//vname )

      nset = 0
      nmiss = 0
      DO i = 1, Mesh % NumberOfNodes
        j = i
        IF ( ASSOCIATED( Var % Perm ) ) THEN
          IF ( i > SIZE( Var % Perm ) ) CYCLE
          j = Var % Perm(i)
        END IF
        IF ( j <= 0 .OR. j > SIZE( Var % Values ) ) CYCLE
        k = FindGid( gids, n, NodeGid(i) )
        IF ( k > 0 ) THEN
          Var % Values(j) = vals(k)
          IF ( ASSOCIATED( Var % PrevValues ) ) THEN
            DO l = 1, SIZE( Var % PrevValues, 2 )
              Var % PrevValues(j,l) = vals(k)
            END DO
          END IF
          nset = nset + 1
        ELSE
          nmiss = nmiss + 1
        END IF
      END DO

      WRITE( Message, '(A,A,A,I0,A,I0,A)' ) 'Read "', vname, '": set ', nset, &
          ' nodes, ', nmiss, ' not in file'
      CALL Info( Caller, Message, Level = 4 )
      IF ( Strict .AND. nmiss > 0 ) THEN
        CALL Fatal( Caller, 'State file does not cover all nodes of "'//vname// &
            '" (mesh/partitioning differs from the one the state was extracted on?)' )
      END IF
    END DO
    CLOSE( u )
  END SUBROUTINE ReadState


  SUBROUTINE WriteState( fname )
    CHARACTER(*) :: fname
    INTEGER :: u, ios, nvar, nrec, kv, i, j, c, cnt
    INTEGER(KIND=8), ALLOCATABLE :: gids(:)
    REAL(KIND=dp), ALLOCATABLE :: vals(:)
    CHARACTER(:), ALLOCATABLE :: vname, oname
    CHARACTER(LEN=MAX_NAME_LEN) :: names(100)
    TYPE(Variable_t), POINTER :: Var
    LOGICAL :: WriteCoords, Found
    CHARACTER(LEN=16) :: cname

    nvar = 0
    DO kv = 1, 100
      vname = ListGetString( Params, 'State Variable '//I2S(kv), Found )
      IF ( .NOT. Found ) EXIT
      Var => VariableGet( Mesh % Variables, vname, ThisOnly = .TRUE. )
      IF ( .NOT. ASSOCIATED( Var ) ) THEN
        CALL Warn( Caller, 'Variable to write not found, skipping: '//vname )
        CYCLE
      END IF
      IF ( Var % DOFs /= 1 ) THEN
        CALL Warn( Caller, 'Only scalar variables can be written, skipping: '//vname )
        CYCLE
      END IF
      nvar = nvar + 1
      names(nvar) = vname
    END DO

    WriteCoords = ListGetLogical( Params, 'Write Coordinates', Found )
    nrec = nvar
    IF ( WriteCoords ) nrec = nrec + 3

    oname = fname//'.'//I2S(ParEnv % MyPE)
    OPEN( NEWUNIT=u, FILE=oname, ACCESS='STREAM', FORM='UNFORMATTED', &
        STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
    IF ( ios /= 0 ) CALL Fatal( Caller, 'Cannot open for writing: '//oname )
    WRITE( u ) ESS_MAGIC, ESS_VERSION, nrec

    ALLOCATE( gids(Mesh % NumberOfNodes), vals(Mesh % NumberOfNodes) )

    DO kv = 1, nvar
      Var => VariableGet( Mesh % Variables, TRIM(names(kv)), ThisOnly = .TRUE. )
      cnt = 0
      DO i = 1, Mesh % NumberOfNodes
        j = i
        IF ( ASSOCIATED( Var % Perm ) ) THEN
          IF ( i > SIZE( Var % Perm ) ) CYCLE
          j = Var % Perm(i)
        END IF
        IF ( j <= 0 .OR. j > SIZE( Var % Values ) ) CYCLE
        cnt = cnt + 1
        gids(cnt) = NodeGid(i)
        vals(cnt) = Var % Values(j)
      END DO
      CALL WriteRecord( u, TRIM(names(kv)), cnt, gids, vals )
    END DO

    IF ( WriteCoords ) THEN
      DO c = 1, 3
        DO i = 1, Mesh % NumberOfNodes
          gids(i) = NodeGid(i)
          SELECT CASE ( c )
          CASE ( 1 )
            vals(i) = Mesh % Nodes % x(i)
          CASE ( 2 )
            vals(i) = Mesh % Nodes % y(i)
          CASE ( 3 )
            vals(i) = Mesh % Nodes % z(i)
          END SELECT
        END DO
        WRITE( cname, '(A,I0)' ) 'coordinate ', c
        CALL WriteRecord( u, TRIM(cname), Mesh % NumberOfNodes, gids, vals )
      END DO
    END IF

    CLOSE( u )
    CALL Info( Caller, 'Wrote state file: '//oname, Level = 4 )
  END SUBROUTINE WriteState


  SUBROUTINE WriteRecord( u, name, cnt, gids, vals )
    INTEGER :: u, cnt
    CHARACTER(*) :: name
    INTEGER(KIND=8) :: gids(:)
    REAL(KIND=dp) :: vals(:)
    WRITE( u ) INT( LEN(name), 4 )
    WRITE( u ) name
    WRITE( u ) INT( cnt, 8 )
    IF ( cnt > 0 ) THEN
      WRITE( u ) gids(1:cnt)
      WRITE( u ) vals(1:cnt)
    END IF
  END SUBROUTINE WriteRecord

END SUBROUTINE StateIO


!------------------------------------------------------------------------------
! Placeholder solver: owns exported variables (frozen masks, Zb) that are set
! only by StateIO / initial conditions and never recomputed.
!------------------------------------------------------------------------------
SUBROUTINE ESSNoOp( Model, Solver, dt, TransientSimulation )
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t), TARGET :: Solver
  REAL(KIND=dp) :: dt
  LOGICAL :: TransientSimulation
  RETURN
END SUBROUTINE ESSNoOp
