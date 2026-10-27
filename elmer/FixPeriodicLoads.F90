!------------------------------------------------------------------------------
!> FixPeriodicLoads: a periodic copy of a nodal field (default: the Stokes loads "Flow Solution Loads").
!>
!> Why: with an implicit periodic BC, Elmer adds the image node's row into the master node's row of the
!> matrix and RHS, including the saved bulk copy that "Calculate Loads" uses (SetPeriodicBoundariesPass1), and
!> then replaces only the live image row with the constraint. So the computed loads are
!>     master: its own load + the image's load (the total of the periodic node),
!>     image:  its own, one-sided load (it includes the force across the periodic seam).
!> With the default mode "folded" every member of a periodic group (pairs, and corner chains) gets the group
!> total / members (PeriodicUtils). Exact for a conforming periodic mesh whose members own equal boundary areas.
!>
!> The result goes into a new variable by default (same Perm, same DOFs, components "<target> 1..n"), so
!> solvers that expect the raw loads keep them; Target Variable = the source works in place (e.g. a mask).
!> Run it after the solver that computes the field and before whatever uses it.
!>
!> Keywords (Solver section):
!>   Loads Variable  = String "Flow Solution Loads"           (default; "Source Variable" also accepted)
!>   Target Variable = String "Periodic Flow Solution Loads"  (default: "Periodic " // source)
!>   Periodic Mode   = String "folded"                        (folded | split | mean | fold | min | max)
!------------------------------------------------------------------------------
SUBROUTINE FixPeriodicLoads( Model, Solver, dt, Transient )
!------------------------------------------------------------------------------
  USE DefUtils
  USE PeriodicUtils
  IMPLICIT NONE
  TYPE(Model_t) :: Model
  TYPE(Solver_t) :: Solver
  REAL(KIND=dp) :: dt
  LOGICAL :: Transient
!------------------------------------------------------------------------------
  TYPE(Mesh_t), POINTER :: Mesh
  TYPE(ValueList_t), POINTER :: Params
  TYPE(Variable_t), POINTER :: Src, Tgt
  INTEGER :: dofs, npairs, norphan
  REAL(KIND=dp), POINTER :: NewValues(:)
  CHARACTER(LEN=MAX_NAME_LEN) :: sname, tname, pmode
  CHARACTER(*), PARAMETER :: Caller = 'FixPeriodicLoads'
  LOGICAL :: Found
!------------------------------------------------------------------------------
  Params => GetSolverParams()
  Mesh => GetMesh()

  sname = ListGetString( Params, 'Loads Variable', Found )
  IF ( .NOT. Found ) sname = ListGetString( Params, 'Source Variable', Found )
  IF ( .NOT. Found ) sname = 'Flow Solution Loads'
  tname = ListGetString( Params, 'Target Variable', Found )
  IF ( .NOT. Found ) tname = 'Periodic '//TRIM(sname)
  pmode = ListGetString( Params, 'Periodic Mode', Found )
  IF ( .NOT. Found ) pmode = 'folded'

  Src => VariableGet( Mesh % Variables, TRIM(sname), ThisOnly = .TRUE. )
  IF ( .NOT. ASSOCIATED( Src ) ) THEN
    CALL Warn( Caller, 'No variable "'//TRIM(sname)//'" (yet): nothing done' )
    RETURN
  END IF
  IF ( .NOT. ASSOCIATED( Src % Perm ) ) CALL Fatal( Caller, '"'//TRIM(sname)//'" has no Perm' )
  dofs = Src % DOFs

  Tgt => VariableGet( Mesh % Variables, TRIM(tname), ThisOnly = .TRUE. )
  IF ( .NOT. ASSOCIATED( Tgt ) ) THEN
    ! the same size as the source (VariableAddVector would size it by MAXVAL(Perm), which can differ)
    ALLOCATE( NewValues( SIZE( Src % Values ) ) )
    NewValues = 0.0_dp
    CALL VariableAddVector( Mesh % Variables, Mesh, Solver, TRIM(tname), dofs, Values = NewValues, &
        Perm = Src % Perm )
    Tgt => VariableGet( Mesh % Variables, TRIM(tname), ThisOnly = .TRUE. )
    CALL Info( Caller, 'Created "'//TRIM(tname)//'" with '//I2S(dofs)//' components', Level=5 )
  END IF
  IF ( .NOT. ASSOCIATED( Tgt, Src ) ) THEN
    IF ( SIZE( Tgt % Values ) /= SIZE( Src % Values ) ) &
        CALL Fatal( Caller, '"'//TRIM(tname)//'" and "'//TRIM(sname)//'" differ in size' )
    Tgt % Values = Src % Values
  END IF

  CALL PeriodicSymmetrize( Model, Tgt % Values, Src % Perm, dofs, TRIM(pmode), npairs, norphan )
  CALL Info( Caller, '"'//TRIM(tname)//'" ('//TRIM(pmode)//'): '//I2S(npairs)//' periodic pairs', Level=4 )
!------------------------------------------------------------------------------
END SUBROUTINE FixPeriodicLoads
!------------------------------------------------------------------------------
