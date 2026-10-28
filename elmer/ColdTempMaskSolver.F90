! Copied unchanged from ElmerStreams SRC/ColdTempMaskSolver.f90 (thermal = "elmerstreams" in ESS).
! Solver for creating a maks that identifies where the temperate and cold BCs should be applied.
! Inspired by the GroundedSolver.f90 used to make the GroundedMask.
!
! 7th Nov 2025
!
SUBROUTINE ColdTempMaskSolver(Model,Solver,dt,TransientSimulation)
!------------------------------------------------------------------------
!************************************************************************
!
!
!  For the bottom ice surface (ice-bedrock interface), creates and updates a mask which may be equal to +1 or -1.
!
! ColdTempMask = +1 if TEMPERATE BC should be applied
!              = -1 if COLD BC should be applied
!
! The value of the mask is determined based on the mask used for and the resulting temperature and water sheet thickness solutions from the previous timestep/iteration (?)
! If the previous value of the mask lead to a contradiction in the solution (T>=Tm or hw>0), the sign of the mask is switched.
!
! ARGUMENTS:
!
!  TYPE(Model_t) :: Model,  
!     INPUT: All model information (mesh, materials, BCs, etc...)
!
!  TYPE(Solver_t) :: Solver
!     INPUT: Linear & nonlinear equation solver options
!
!  REAL(KIND=dp) :: dt,
!     INPUT: Timestep size for time dependent simulations
!
!  LOGICAL :: TransientSimulation
!     INPUT: Steady state or transient simulation
!
!**************************************************************************

  Use DefUtils

  IMPLICIT NONE

  !----------------------------------------------------

  TYPE(Solver_t) :: Solver
  TYPE(Model_t) :: Model

  REAL(KIND=dp) :: dt
  LOGICAL :: TransientSimulation

! ---------------------------
! Local variables
! ---------------------------

  TYPE(Element_t), POINTER :: Element
  TYPE(ValueList_t), POINTER :: Material, SolverParams
  TYPE(Variable_t), POINTER :: PointerToVariable
  TYPE(Nodes_t), SAVE :: Nodes
  TYPE(Mesh_t), POINTER :: Mesh

  LOGICAL :: Found, AllocationsDone = .FALSE.

  INTEGER :: i, n, t, j, dim, mn, istat

  REAL(KIND=dp) :: toler
  REAL(KIND=dp), ALLOCATABLE :: Tm(:)

  SAVE AllocationsDone, dim, SolverName, toler, Tm

  TYPE(Variable_t), POINTER :: TempSol, HwSol  ! These all 'point' to memory with different names elsewhere
  REAL(KIND=dp), POINTER :: ColdTempValues(:), ColdTempPrev(:), TempValues(:), TempPrev(:), HwValues(:), HwPrev(:,:)
  INTEGER, POINTER :: ColdTempPerm(:), TempPerm(:), HwPerm(:)   ! Used to match up node number with solution value at that node (not obvious due to how Elmer stores values)

  CHARACTER(LEN=MAX_NAME_LEN) :: SolverName = 'ColdTempMaskSolver'

  PointerToVariable => Solver % Variable           ! Points to the ColdTempMask variable
  ColdTempPerm => PointerToVariable % Perm         ! Identifier for which of the variable values is the value for a given node
  ColdTempValues => PointerToVariable % Values     ! Values of ColdTempMask
  ColdTempPrev => PointerToVariable % PrevValues(:,1)   ! Values from previous timestep (or iteration?)

  Mesh => Solver % Mesh


!------------------------------------------
! Allocate some permenant storage:
!------------------------------------------
  IF ( (.NOT. AllocationsDone) .OR. Solver % Mesh % Changed ) THEN
    dim = CoordinateSystemDimension()
    mn = Solver % Mesh % MaxElementNodes
    IF (AllocationsDone) THEN
      DEALLOCATE(Tm)
    END IF

    ALLOCATE(Tm(mn), STAT = istat)

    IF ( istat /= 0 ) THEN
        CALL FATAL( SolverName, 'Memory allocation error' )
      ELSE
        CALL INFO( SolverName, 'Memory allocation done', level=1 )
      END IF

      AllocationsDone = .TRUE.

    END IF


  HwSol => VariableGet( Mesh % Variables, 'Water Sheet Thickness' )
  IF ( ASSOCIATED( HwSol ) ) THEN
    HwPerm     => HwSol % Perm
    HwValues => HwSol % Values
    HwPrev => HwSol % PrevValues!(:,1)    ! Values from previous timestep? Or iteration?
  ELSE
  CALL FATAL(SolverName, "Pointer to Water Sheet Thickness not associated")
  END IF

  TempSol => VariableGet( Mesh % Variables, 'Temperature' )
  IF ( ASSOCIATED( TempSol ) ) THEN
    TempPerm     => TempSol % Perm
    TempValues => TempSol % Values
    TempPrev => TempSol % PrevValues(:,1)    ! Values from previous timestep? Or iteration?
  ELSE
  CALL FATAL(SolverName, "Pointer to Temperature not associated")
  END IF

  SolverParams => GetSolverParams()
  toler = GetConstReal(SolverParams, 'Toler', Found)
  IF (.NOT.Found) THEN
     CALL FATAL(SolverName, 'No tolerance given for the ColdTempMask.')
  END IF

CALL INFO( SolverName, 'setting vals', level=1 )
! ---------------------------------------------
! Set ColdTempMask values
! ---------------------------------------------

  DO t = 1, Solver % NumberOfActiveElements
    Element => GetActiveElement(t)
    n = GetElementNOFNodes()

    CALL GetElementNodes(Nodes)

    Material => GetMaterial()

    Tm(1:n)=ListGetReal( Material, 'Tm', n, Element % NodeIndexes, Found)   ! melting temperature


    DO i = 1,n
      j = Element % NodeIndexes(i)
      
      IF (ParEnv % myPe .NE. Element % partIndex) CYCLE


      ! IF (j==0) CYCLE

      ! If we assumed the node was TEMPERATE (i.e. mask value = +1)....
      IF (ColdTempPrev(ColdTempPerm(j)) .GE. 0.0_dp) THEN 
       	IF (HwPrev(HwPerm(j),1) .LE. 0) THEN
          ColdTempValues(ColdTempPerm(j)) = -1.0_dp     ! ... switch the mask to -1 (use COLD BC) for the next iteration/timestep (?) ...
        ELSE
          ColdTempValues(ColdTempPerm(j)) = 1.0_dp    ! ... otherwise, keep as +1 (use TEMP BC again).
        END IF

      ! If we assumed the node was COLD (i.e. mask value = -1)...
      ELSE IF (ColdTempPrev(ColdTempPerm(j)) .LE. 0.0_dp) THEN
        IF ( (TempPrev(TempPerm(j)) - Tm(i)) .GE. toler) THEN    ! ... and we calculated a temperature ABOVE THE MELTING POINT...
          ColdTempValues(ColdTempPerm(j)) = 1.0_dp                ! ... switch the mask to +1 (use TEMP BC) got the next iteration/timestep (?) ...
        ELSE
          ColdTempValues(ColdTempPerm(j)) = -1.0_dp               ! ... otherwise, keep as -1 (use cOLD BC again).
        END IF
      END IF
      IF (ColdTempValues(ColdTempPerm(j)) .LE. 0) THEN
        HwValues(HwPerm(j))  = 0.0001_dp
      END IF
    END DO

    
  END DO

  
  !IF ( ParEnv % PEs > 1) CALL ParallelSumVector( Solver % Matrix, ColdTempValues, 1)

CALL INFO(SolverName,'Done',level=1)

END SUBROUTINE ColdTempMaskSolver
