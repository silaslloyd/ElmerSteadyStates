! Copied unchanged from ElmerStreams SRC/sheetsolverhw.f90 (Marine_Term: Procedure "../SRC/SheetSolverhw").
!/*****************************************************************************/
! *
! *  Elmer, A Finite Element Software for Multiphysical Problems
! *
! *  Copyright 1st April 1995 - , CSC - IT Center for Science Ltd., Finland
! * 
! *  This library is free software; you can redistribute it and/or
! *  modify it under the terms of the GNU Lesser General Public
! *  License as published by the Free Software Foundation; either
! *  version 2.1 of the License, or (at your option) any later version.
! *
! *  This library is distributed in the hope that it will be useful,
! *  but WITHOUT ANY WARRANTY; without even the implied warranty of
! *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
! *  Lesser General Public License for more details.
! * 
! *  You should have received a copy of the GNU Lesser General Public
! *  License along with this library (in file ../LGPL-2.1); if not, write 
! *  to the Free Software Foundation, Inc., 51 Franklin Street, 
! *  Fifth Floor, Boston, MA  02110-1301  USA
! *
! *****************************************************************************/
!/*****************************************************************************/
! *
! * A prototype solver for advection-diffusion-reaction equation,
! * This equation is generic and intended for education purposes
! * but may also serve as a starting point for more complex solvers.
! *
! *  Web:     http://www.csc.fi/elmer
! *  Address: CSC - IT Center for Science Ltd.
! *           Keilaranta 14
! *           02101 Espoo, Finland 
! *
! *****************************************************************************/

!------------------------------------------------------------------------------
SUBROUTINE SheetSolverhw( Model,Solver,dt,TransientSimulation )
!------------------------------------------------------------------------------
  USE DefUtils

  IMPLICIT NONE
!------------------------------------------------------------------------------
  TYPE(Solver_t) :: Solver
  TYPE(Model_t) :: Model
  REAL(KIND=dp) :: dt
  LOGICAL :: TransientSimulation
!------------------------------------------------------------------------------
! Local variables (i.e. for a given element)
!------------------------------------------------------------------------------
  TYPE(Element_t),POINTER :: Element
  REAL(KIND=dp) :: Norm
  INTEGER :: i, j, m, n, n_nei, nb, nd, t, active,passive, istat, correctedvalues  , num_cold !, n = number of nodes (given element), n = number of degrees of freedom (given element), nd = number of DOFs on the boundary (given element), t = index for..., active = number of active element
  INTEGER :: iter, maxiter  ! nonlinear iteration number, max number of nonlinear iterations
  LOGICAL :: Found, Newton, BulkUpdate, RHSUpdate, AllocationsDone = .FALSE.
  TYPE(Mesh_t), POINTER :: Mesh 
  TYPE(ValueList_t), POINTER :: BodyForce, Material, SolverParams 
!------------------------------------------------------------------------------
  TYPE(Variable_t), POINTER :: hwSol, qwSol, coldtempmask, nodalcoords1 ! These all 'point' to memory with different names elsewhere
  REAL(KIND=dp), POINTER :: hw(:), qw(:), ctvals(:), xvals(:)
  REAL(KIND=dp), ALLOCATABLE :: hwOld(:)  ! copy of values to retain when pointer is overwritten by new iteration values
  INTEGER, POINTER :: hwPerm(:), qwPerm(:), ctperm(:), Xperm(:)  ! Used to match up node number with solution value at that node (not obvious due to how Elmer stores values)
  INTEGER, ALLOCATABLE :: hwOldPerm(:), coordinationnumber(:), numberofpassiveneighbours(:)  ! copy of values to retain when pointer is overwritten by new iteration values
  TYPE(Nodes_t) :: ElementNodes
  INTEGER :: dim, qwNDOFs, k, rankA, rankM, dimsheet
  !REAL(KIND=dp), ALLOCATABLE :: nodalhw(:), dhwdx(:,:), gradPhi0(:,:), dBasisdx(:,:), nodalhwOld(:), dhwdxOld(:,:), nodalqw(:,:)
  !REAL(KIND=dp), ALLOCATABLE :: DensityWater(:), LatentHeat(:), Phi0(:), HydraulicConductivity(:), EffectivePressure(:)
  !REAL(KIND=dp), ALLOCATABLE :: dEffectivePressuredh(:), ddEffectivePressuredh(:), dHydraulicConductivitydx(:)
  !REAL(KIND=dp), ALLOCATABLE :: q0(:,:), qh(:,:), QQh(:)
  REAL(KIND=dp), ALLOCATABLE :: MASS(:,:), STIFF(:,:), LOAD(:), FORCE(:)

  CHARACTER(LEN=MAX_NAME_LEN) :: SolverName

  SAVE hwOld, hwOldPerm, MASS, STIFF, LOAD, FORCE, coordinationnumber, numberofpassiveneighbours

!PointerToSolver => Solver    ! https://fortran-lang.discourse.group/t/understanding-fortran-pointers/1142

  SolverParams => GetSolverParams()    ! Access information (keywords) from the relevant solver section in the sif

  SolverName = 'SheetSolverhw'

! Details of mesh - IS THIS NEEDED HERE?
  Mesh => Solver % Mesh
  dim = Mesh % MeshDim     ! 1, 2 or 3 D
  m = Mesh % NumberOfNodes      ! Number of nodes in the mesh
  n = Mesh % MaxElementNodes    ! Maximum number of nodes that there could be in a single element

! Allocate some permenant storage
  IF ( .NOT. AllocationsDone .OR. Mesh % Changed ) THEN

    IF ( AllocationsDone ) THEN
      DEALLOCATE( hwOld, hwOldPerm )
    END IF

    ALLOCATE( hwOld(m), hwOldPerm(m), MASS(n,n), STIFF(n,n), LOAD(n), FORCE(n), &
         coordinationnumber(m), numberofpassiveneighbours(m), STAT=istat )

    IF ( istat /= 0 ) THEN
      CALL FATAL( 'SheetSolver', 'Memory allocation error' )
    ELSE
      CALL INFO( 'SheetSolver', 'Memory allocation done', level=1 )
    END IF

    AllocationsDone = .TRUE.

  END IF

! Point to water sheet thickness solution: the values and how those values match up with the nodes
  hwSol => Solver % Variable
  hwPerm  => hwSol % Perm    ! Identifier for which of the values (hw) is the value for a given node
  hw => hwSol % Values   ! Values of the water sheet thickness
  hwOld = hw  ! a copy of the hw values from the previous iteration so I can use them after the FEM system solve (during which the pointer values will be updated)
  hwOldPerm = hwPerm

! Point to water flux solution
  qwSol => VariableGet( Mesh % Variables, 'Water Flux' )
  IF ( ASSOCIATED( qwSol ) ) THEN
    qwPerm => qwSol % Perm
    qwNDOFs = qwSol % DOFs  ! number of vector components of the water flux (i.e. 3) 
    qw => qwSol % Values  ! all components of
  ELSE
  CALL FATAL("sheetsolverhw", "Pointer to Water Flux not associated")
  END IF

  coldtempmask => VariableGet( Mesh % Variables, 'ColdTempMask' )
  IF ( ASSOCIATED( coldtempmask ) ) THEN
        ctperm     => coldtempmask % Perm
        ctvals => coldtempmask % Values
  ELSE
    CALL FATAL(SolverName, "Pointer to CT Mask not associated")
  END IF

  nodalcoords1 => VariableGet( Mesh % Variables, 'nodalcoords 1' )
  IF ( ASSOCIATED( coldtempmask ) ) THEN
        Xperm     => nodalcoords1 % Perm
        xvals => nodalcoords1 % Values
  ELSE
    CALL FATAL(SolverName, "Pointer to CT Mask not associated")
  END IF
  CALL DefaultStart()
  
  maxiter = ListGetInteger( GetSolverParams(),&
      'Nonlinear System Max Iterations',Found,minv=1)
  IF(.NOT. Found ) maxiter = 1

  ! Nonlinear iteration loop:
  !--------------------------
  DO iter=1,maxiter
    CALL Info('SheetSolverhw','Sheet solver iteration: '//I2S(iter))

    !Newton =  GetNewtonActive()  ! logical that is 1 if Newton iterations are to be used (Picard otherwise)
    Newton = .False.

    !WRITE(*,*) Newton
    ! System assembly:
    !----------------
    CALL DefaultInitialize()
    
    ! Build matrix arising from the bulk elements
    ! ------------------
    Active = GetNOFActive()   ! number of active elements (i.e. not passive)
    !WRITE(*,*) Active, " active elements"

    ! per-node counter initialization
    coordinationnumber=0
    numberofpassiveneighbours=0
    
    DO t=1,Active   ! for each active element..
      Element => GetActiveElement(t)   ! information about that element
      !WRITE(*,*) "___________________"
      !WRITE(*,*) "ELEMENT", t
      num_cold = 0._dp ! REMOVED, AS UNUSED IN THIS CONTEXT
      !IF (ParEnv % myPe .NE. Element % partIndex) CYCLE
      
      n  = Element % TYPE % NumberOfNodes !GetElementNOFNodes()        ! number of nodes
      nd = GetElementNOFDOFs()         ! number of degrees of freedom (DOF)
      nb = GetElementNOFBDOFs()        ! number of BUBBLES DOFs (NOT boundary DOFs)
      
      dimsheet = Element % TYPE % DIMENSION

      ! Set per-node counters for coordination number and number of passive elements for later post-processing step
      coordinationnumber(Element % NodeIndexes(1:n)) = coordinationnumber(Element % NodeIndexes(1:n)) + 1
      DO i=1, n   ! ... for each node within the element (LOCAL nodal index)
        j = Element % NodeIndexes(i) 
        IF (ctvals(ctperm(j)) .LE. 0) THEN
          num_cold = num_cold + 1
        END IF
      END DO

      IF (num_cold > 0._dp) THEN
      !IF (CheckPassiveElement(Element)) THEN
        numberofpassiveneighbours(Element % NodeIndexes(1:n)) = numberofpassiveneighbours(Element % NodeIndexes(1:n)) + 1
      END IF

     !WRITE(*,*) "___________________"

      CALL LocalMatrix(  Element, n, nd, dim, dimsheet, hw(hwPerm(Element % NodeIndexes(1:n))), ctperm, ctvals)
      
    
    END DO

    !CALL DefaultFinishBulkAssembly()
    ! If the second and third entries are TRUE logicals, the matrices and force vector will be saved
    BulkUpdate = .TRUE.   ! mass and stiffness matrices are saved if true
    RHSUpdate = .TRUE.    ! force vector is saved if true
    CALL DefaultFinishBulkAssembly(Solver,BulkUpdate,RHSUpdate)
    
    
    CALL DefaultFinishBoundaryAssembly()
    CALL DefaultFinishAssembly()
    CALL DefaultDirichletBCs()
    ! And finally, solve:
    !--------------------
    Norm = DefaultSolve() 
    !IF ( ParEnv % PEs > 1) CALL ParallelSumVector( Solver % Matrix, hw, 2)

   IF( DefaultConverged() ) THEN
    EXIT    
   END IF

  END DO

  ! Correct orphan "left-over" temperate nodes with no connection to the hydrologic system to cold
  correctedvalues = 0
  DO I= 1,Model % Mesh % NumberOfNodes
    IF (coordinationnumber(I) > 0 .AND. ( coordinationnumber(I) .EQ. numberofpassiveneighbours(I))) THEN
      ctvals(ctperm(I)) = -1.0
    END IF
  END DO
  IF ( ParEnv % PEs > 1) CALL ParallelSumVector( Solver % Matrix, ctvals, 2)

  !PRINT *, ParEnv % myPe, ": corrected", correctedvalues, " nodes from temperate to cold"
  
!---------------------------------------------------
! Calculate water flux 
!---------------------------------------------------
! Do this using whichever form was used for the linearisation (i.e. Newton or Picard)
DO t=1, Solver % NumberOfActiveElements   ! for each active element...
  Element => GetActiveElement(t,Solver)
  IF (ParEnv % myPe .NE. Element % partIndex) CYCLE

  n = GetElementNOFNodes(Element)   ! number of nodes in element
  dimsheet = Element % TYPE % DIMENSION

  CALL GetElementNodes( ElementNodes )   ! ElementNodes = nodal coordinates (x,y,z) for nodes in the element

  !CALL GetParameters(Element, Material, n, DensityWater, LatentHeat, Phi0, EffectivePressure, HydraulicConductivity, &
  !  dEffectivePressuredh, ddEffectivePressuredh, dHydraulicConductivitydx) ! Get parameter values (at nodes) from sif file
  
  ! Nodal values of water sheet thickness hw from CURRENT (just calculated) and PREVIOUS iteration
  !DO i=1, n   ! ... for each node within the element (LOCAL nodal index)
  !  j = Element % NodeIndexes(i)  ! almost GLOBAL nodal index (but still need to apply perm)
  !  nodalhw(j) = hw(hwPerm(j))   ! hw at CURRENT iteration (just calculated)
  !  nodalhwOld(j) = hwOld(hwOldPerm(j))   ! hw at PREVIOUS iteration
  !END DO

  !CALL CalculateWaterFluxComponents(Element, n, dimsheet, nodalhwOld, q0, qh, QQh)

  !CALL CalculateWaterFlux(Element, n, dimsheet, hw(hwPerm(Element % NodeIndexes(1:n))), &
  !  hwOld(hwOldPerm(Element % NodeIndexes(1:n))), qw, qwPerm)

END DO 

CALL DefaultFinish()
 
CONTAINS

! Calculate the linearised components of the water flux (used in the local matrices)
! ------------------------------------------------------------------------------------------
  SUBROUTINE CalculateWaterFluxComponents(Element, n, dimsheet, nodalhw, q0, qh, QQh)
! ------------------------------------------------------------------------------------------
    REAL(KIND=dp) :: nodalhw(n), dhwdx(n,dimsheet), gradPhi0(n,dimsheet), dBasisdx(n,dimsheet)
    REAL(KIND=dp) :: DensityWater(n), LatentHeat(n), Phi0(n), HydraulicConductivity(n), EffectivePressure(n)
    REAL(KIND=dp) :: dEffectivePressuredh(n), ddEffectivePressuredh(n), dHydraulicConductivitydx(n)
    REAL(KIND=dp) :: q0(n,dimsheet), qh(n,dimsheet), QQh(n)

    INTEGER :: i, j, n, dimsheet
    
    TYPE(Element_t), POINTER :: Element
    TYPE(ValueList_t), POINTER :: Material

    CALL GetElementNodes( ElementNodes )
     
    ! -----------------------------------------------------------
    ! Access parameter values from material section of sif file (nodal values)
    ! -----------------------------------------------------------
    CALL GetParameters(Element, Material, n, DensityWater, LatentHeat, Phi0, EffectivePressure, HydraulicConductivity, &
    dEffectivePressuredh, ddEffectivePressuredh, dHydraulicConductivitydx)

    ! --------------------------------------------------------------------------------
    ! Calculate the linearised water flux coefficients at each node within the element in question.
    ! --------------------------------------------------------------------------------
    DO i=1, n   ! ... for each node within the element (LOCAL nodal index)
      j = Element % NodeIndexes(i)  ! almost GLOBAL nodal index (but still need to apply perm)

      ! Nodal gradients of water sheet thickness hw and Phi0 (at previous iteration)
      ! (NOTE: dimsheet (2) instead of dim(3) is used for the gradients to avoid nonsensical 
      ! z derivatives on the 2D surface that the hydrology solver is applied on.)
      !-------------------------------------------------------
      dhwdx(i,1:dimsheet) = nodalhw(i) * dBasisdx(i,1:dimsheet)   ! grad(hw)
      gradPhi0(i,1:dimsheet) = Phi0(i) * dBasisdx(i,1:dimsheet)  ! where N = Phi0 - Phi
      !WRITE(*,*) 'dhwdx', dhwdx(i,1:dimsheet)
      !WRITE(*,*) 'gradPhi0', gradPhi0(i,1:dimsheet)
      !WRITE(*,*) 'Phi0', Phi0(i)
      !WRITE(*,*) 'dHydraulicConductivitydx', dHydraulicConductivitydx(i)
      !WRITE(*,*) 'HydraulicConductivity', HydraulicConductivity(i)
      !WRITE(*,*) 'dEffectivePressuredh', dEffectivePressuredh(i)
      !WRITE(*,*) 'EffectivePressure', EffectivePressure(i)
      !WRITE(*,*) 'ddEffectivePressuredh', ddEffectivePressuredh(i)
      !WRITE(*,*) 'hw', nodalhw(i)
      !WRITE(*,*) 'dBasisdx', dBasisdx(i,1:dim)

      ! Coefficient of hw in flux linearisation
      ! ------------------------------------------
      IF (Newton) THEN
        qh(i,1:dimsheet) = dHydraulicConductivitydx(i)*dEffectivePressuredh(i)*(gradPhi0(i,1:dimsheet) - &
          dEffectivePressuredh(i)*dhwdx(i,1:dimsheet)) - HydraulicConductivity(i)*ddEffectivePressuredh(i)*dhwdx(i,1:dimsheet)
      ELSE
        qh(i,1:dimsheet) = 0
      END IF
      !WRITE(*,*) 'qh', qh(i,1:dimsheet)

      ! Coefficient of grad(hw) in flux linearisation
      ! ----------------------------------------------
      IF (Newton) THEN
        QQh(i) = -HydraulicConductivity(i)*dEffectivePressuredh(i)
      ELSE
        QQh(i) = -HydraulicConductivity(i)*dEffectivePressuredh(i)
      END IF
      !WRITE(*,*) 'QQh', QQh(i)

      ! Order 1 term in flux linearisation
      ! ------------------------------------
      IF (Newton) THEN
        q0(i,1:dimsheet) = HydraulicConductivity(i)*gradPhi0(i,1:dimsheet) - qh(i,1:dimsheet)
      ELSE
        q0(i,1:dimsheet) = HydraulicConductivity(i)*gradPhi0(i,1:dimsheet)
      END IF
      !WRITE(*,*) 'q0', q0(i,1:dimsheet)
    END DO

! ----------------------------------------------
  END SUBROUTINE CalculateWaterFluxComponents
! ----------------------------------------------

! Calculate the water flux from the linearised components and the hw solutions at the
! current and previous iterations
! -------------------------------------------------------------------------------
  SUBROUTINE CalculateWaterFlux(Element, n, dimsheet, nodalhw, nodalhwOld, qw, qwPerm)
! -------------------------------------------------------------------------------
!   Element = info about the element
!   n = number of nodes in the element
!   dimsheet = dimension of the element (e.g. 2D)
!   nodalhw = values of water sheet thickness on the nodes from the current iteration (just calculated), 
!             indexed by the local nodal indices
!   nodalhwOld = values of water sheet thickness on the nodes from the previous iteration, 
!             indexed by the local nodal indices
!   qw = nodal values of the water flux, which this subroutine calculates (pointer)
!   qwPerm = permutation for mapping local to global nodal indices (pointer)
! ---------------------------------------------------------------------------------
  
    REAL(KIND=dp) :: nodalhw(n), dhwdx(n,dimsheet), gradPhi0(n,dimsheet), dBasisdx(n,dimsheet)
    REAL(KIND=dp) :: nodalhwOld(n), dhwdxOld(n,dimsheet), nodalqw(n,dimsheet)
    REAL(KIND=dp) :: q0(n,dimsheet), qh(n,dimsheet), QQh(n)

    REAL(KIND=dp), POINTER :: qw(:)
    INTEGER, POINTER :: qwPerm(:)

    INTEGER :: i, j, n, dimsheet
    
    TYPE(Element_t), POINTER :: Element
    TYPE(ValueList_t), POINTER :: Material

    CALL GetElementNodes( ElementNodes )

    ! Calculate the coefficients of hw, grad(hw) and the constant term in the linearisation
    ! used for the water flux qw in the FEM system (e.g. Newton or Picard)
    ! ---------------------------------------------------------------------------------
    CALL CalculateWaterFluxComponents(Element, n, dimsheet, nodalhw, q0, qh, QQh)

    ! Loop over elements to compute the water flux
    ! --------------------------------------------------
    DO i=1, n   ! ... for each node within the element (LOCAL nodal index)
      j = Element % NodeIndexes(i)  ! almost GLOBAL nodal index (but still need to apply perm)
  
      ! Calculate the derivatives of the old and current water sheet thickness hw
      ! ---------------------------------------------------------------------------
      dhwdx(i,1:dimsheet) = nodalhw(i) * dBasisdx(i,1:dimsheet)   ! grad(hw) at current iteration (just calculated)
      dhwdxOld(i,1:dimsheet) = nodalhwOld(i) * dBasisdx(i,1:dimsheet)  ! grad(hw) at previous iteration
  
      ! Reassemlbe all the linearised components to give the water flux qw
      ! ------------------------------------------------------------------------
      ! The pointer qw contains all the components of the Water Flux for all the nodes (global)
      ! stacked together, i.e qx1, qy1, qz1, qx2, qy2, qz2, ...
      ! So ,to reassmeble:

      DO k = 1,dimsheet
        ! Calculate nodalqw only for the dimensions of the water sheet
        nodalqw(i,k) = q0(i,k) + qh(i,k)*nodalhwOld(i) + QQh(i)*dhwdxOld(i,k)  ! value of each component (k) of qw at each node (i) within the element
      END DO
      
      DO k = dimsheet+1,qwNDOFs  ! qwNDOFs should be equal to dim (3)
        ! Set the z component of the water flux to zero, since this is meaningless on out 2D surface
        nodalqw(i,k) = 0
      END DO

      DO k = 1,qwNDOFs
        qw((qwPerm(j)-1)*qwNDOFs+k) = nodalqw(i,k)  ! value of each global node (qwPerm(j))
      END DO

    END DO
  
! ----------------------------------------
  END SUBROUTINE CalculateWaterFlux
! ----------------------------------------


! Get parametrs from sif file
! -----------------------------------------------------------------------------
  SUBROUTINE GetParameters(Element, Material, n, DensityWater, LatentHeat, Phi0, EffectivePressure, HydraulicConductivity, &
    dEffectivePressuredh, ddEffectivePressuredh, dHydraulicConductivitydx)
! -----------------------------------------------------------------------------
  REAL(KIND=dp) :: DensityWater(n), LatentHeat(n), Phi0(n), EffectivePressure(n), HydraulicConductivity(n), &
    dEffectivePressuredh(n), ddEffectivePressuredh(n), dHydraulicConductivitydx(n)
  INTEGER :: n   ! number of nodes
  LOGICAL :: Found = .FALSE.
  TYPE(Element_t), POINTER :: Element
  TYPE(ValueList_t), POINTER :: Material
  TYPE(Nodes_t) :: Nodes

  !CALL GetElementNodes( Nodes )

  Material => GetMaterial()

  ! Get parameter values from the Material section of the sif file, using the keywords used there
  DensityWater(1:n) = ListGetReal( Material, 'Density Water', n, Element % NodeIndexes, Found)
  IF(.NOT.Found) THEN
    CALL WARN(SolverName,'Keyword >Density Water< not found in section Constants')
  END IF

  LatentHeat(1:n) = ListGetReal( Material, 'Latent Heat Capacity', n, Element % NodeIndexes, Found)
  IF(.NOT.Found) THEN
    CALL WARN(SolverName,'Keyword >Latent Heat Capacity< not found in section Constants')
  END IF

  Phi0(1:n) = ListGetReal( Material, 'Phi0', n, Element % NodeIndexes, Found)  ! where Effective Pressure = Phi0 - Hydraulic Conductivity
  IF(.NOT.Found) THEN
    CALL WARN(SolverName,'Keyword >Phi0< not found in section Constants')
  END IF

  EffectivePressure(1:n) = ListGetReal( Material, 'Effective Pressure', n, Element % NodeIndexes, Found)
  IF(.NOT.Found) THEN
    CALL WARN(SolverName,'Keyword >Effective Pressure< not found in section Constants')
  END IF

  HydraulicConductivity(1:n) = ListGetReal( Material, 'Hydraulic Conductivity', n, Element % NodeIndexes, Found)
  IF(.NOT.Found) THEN
    CALL WARN(SolverName,'Keyword >Hydraulic Conductivity< not found in section Constants')
  END IF

  dEffectivePressuredh(1:n) = ListGetReal( Material, 'Effective Pressure First Derivative', n, Element % NodeIndexes, Found)
  IF(.NOT.Found) THEN
    CALL WARN(SolverName,'Keyword >Effective Pressure First Derivative< not found in section Constants')
  END IF

  ddEffectivePressuredh(1:n) = ListGetReal( Material, 'Effective Pressure Second Derivative', n, Element % NodeIndexes, Found)
  IF(.NOT.Found) THEN
    CALL WARN(SolverName,'Keyword >Effective Pressure Second Derivative< not found in section Constants')
  END IF

  dHydraulicConductivitydx(1:n) = ListGetReal( Material, 'Hydraulic Conductivity First Derivative', n, Element % NodeIndexes, Found) ! as a function of N, not hw
  IF(.NOT.Found) THEN
    CALL WARN(SolverName,'Keyword >Hydraulic Conductivity First Derivative< not found in section Constants')
  END IF

!-------------------------------------------------------------------------------
  END SUBROUTINE GetParameters
!-------------------------------------------------------------------------------


! Assembly of the matrix entries arising from the bulk elements
!------------------------------------------------------------------------------
  SUBROUTINE LocalMatrix( Element, n, nd, dim, dimsheet, nodalhw, ctperm, ctvals)
!------------------------------------------------------------------------------
    INTEGER :: n, nd, dimsheet, num_cold
    TYPE(Element_t), POINTER :: Element
    TYPE(variable_t), POINTER :: Timevar
!------------------------------------------------------------------------------
    REAL(KIND=dp) :: DensityWater(n), LatentHeat(n), nodalhw(n), Phi0(n), time
    REAL(KIND=dp), POINTER :: ctvals(:)
    INTEGER, POINTER :: ctperm(:)  ! Used to match up node number with solution value at that node (not obvious due to how Elmer stores values)
    REAL(KIND=dp) :: gradPhi0(n,dimsheet), gradN(n,dimsheet), graddNdh(n,dimsheet), HydraulicConductivity(n), EffectivePressure(n)
    REAL(KIND=dp) :: dEffectivePressuredh(n), ddEffectivePressuredh(n), dHydraulicConductivitydx(n)
    REAL(KIND=dp) :: DensityWaterAtIP, LatentHeatAtIP, hwAtIP
    REAL(KIND=dp) :: LoadAtIP, Weight
    REAL(KIND=dp) :: HydraulicConductivityAtIP, EffectivePressureAtIP, dEffectivePressuredhAtIP
    REAL(KIND=dp) :: ddEffectivePressuredhAtIP, dHydraulicConductivitydxAtIP
    REAL(KIND=dp) :: q0(n,dimsheet), qh(n,dimsheet), QQh(n)
    REAL(KIND=dp) :: QQhAtIp
    REAL(KIND=dp) :: dqdhAtIP(dimsheet), q0AtIP(dimsheet), dhwdxAtIP(dimsheet), gradPhi0AtIP(dimsheet), graddNdhAtIP(dimsheet)
    REAL(KIND=dp) :: qhAtIP(dimsheet), gradNAtIP(dimsheet)
    REAL(KIND=dp) :: Basis(nd),dBasisdx(nd,dim),DetJ
    REAL(KIND=dp) :: MASS(nd,nd), STIFF(nd,nd), FORCE(nd), LOAD(n)
    LOGICAL :: Stat,Found
    INTEGER :: i,t,p,q,dim, rankA, rankM
    TYPE(GaussIntegrationPoints_t) :: IP
    TYPE(ValueList_t), POINTER :: BodyForce, Material
    TYPE(Nodes_t) :: Nodes
    SAVE Nodes
!------------------------------------------------------------------------------
      Timevar => VariableGet( Model % Variables,'Time')
      time = TimeVar % Values(1)

    dim = CoordinateSystemDimension()
    dimsheet = Element % TYPE % DIMENSION  ! should be 2 when because the hydrology solver is applied on a 2D boundary

    CALL GetElementNodes( Nodes )
    MASS  = 0._dp
    STIFF = 0._dp
    FORCE = 0._dp
    LOAD = 0._dp
    
    ! Get the basal melt source (from the heat solver) as the source of the hydrology solver
    BodyForce => GetBodyForce()
    IF ( ASSOCIATED(BodyForce) ) &
       Load(1:n) = GetReal( BodyForce,'Melt Source', Found )   ! Supply of water from melting ice
    
    ! Get material property parameter values (at nodes for this element) from sif file
    CALL GetParameters(Element, Material, n, DensityWater, LatentHeat, Phi0, EffectivePressure, HydraulicConductivity, &
      dEffectivePressuredh, ddEffectivePressuredh, dHydraulicConductivitydx)

      
    ! Calculate the linearised components of the water flux (Newton or Picard)
    !---------------------------------------------------------------------------
    !CALL CalculateWaterFluxComponents(Element, n, dimsheet, nodalhw, q0, qh, QQh)

    ! Numerical integration:

    !-----------------------
    IP = GaussPointsAdapt( Element )   ! integration points
    IF( Element % ElementIndex == 1 ) THEN
      CALL Info('SheetSolver','Integration points in 1st element: '//I2S(IP % n),Level=8)
    END IF
    
    num_cold = 0._dp

     DO i=1, n   ! ... for each node within the element (LOCAL nodal index)
        j = Element % NodeIndexes(i)
        IF (ctvals(ctperm(j)) .LE. 0) THEN
          num_cold = num_cold + 1
        END IF
    END DO

    DO t=1,IP % n
      ! Basis function values & derivatives at the integration point:
      !--------------------------------------------------------------
      stat = ElementInfo( Element, Nodes, IP % U(t), IP % V(t), &
              IP % W(t), detJ, Basis, dBasisdx )

      ! The source term at the integration point:
      !------------------------------------------
      LoadAtIP = SUM( Basis(1:n) * LOAD(1:n) )

      ! Water sheet thickness (from previous iteration, I hope) at integration point
      !--------------------------------------------------
      hwAtIP = SUM( nodalhw(1:n) * Basis(1:n) )   ! hw
      dhwdxAtIP = MATMUL( nodalhw(1:n) , dBasisdx(1:n,1:dimsheet))   ! grad(hw)

      !Gradient Values at IPs
      gradPhi0AtIP = MATMUL( Phi0(1:n) , dBasisdx(1:n,1:dimsheet) )  ! where N = Phi0 - Phi
      gradNAtIP = MATMUL( EffectivePressure(1:n) , dBasisdx(1:n,1:dimsheet) )
      graddNdhAtIP = MATMUL( dEffectivePressuredh(1:n) , dBasisdx(1:n,1:dimsheet) )

      !Non Gradient Values at IPs
      HydraulicConductivityAtIP = SUM( HydraulicConductivity(1:n) * Basis(1:n) )

      !IF (num_cold > 0._dp) THEN     
      !    HydraulicConductivityAtIP = 0._dp
      !END IF

      EffectivePressureAtIP = SUM( EffectivePressure(1:n) * Basis(1:n) )
      dEffectivePressuredhAtIP = SUM( dEffectivePressuredh(1:n) * Basis(1:n) )
      dHydraulicConductivitydxAtIP = SUM( dHydraulicConductivitydx(1:n) * Basis(1:n) )
      
      q0AtIP = -HydraulicConductivityAtIP*gradPhi0AtIP

      Weight = IP % s(t) * DetJ
      
      STIFF(1:nd,1:nd) = STIFF(1:nd,1:nd) - Weight * &
              (HydraulicConductivityAtIP)*dEffectivePressuredhAtIP * &
          MATMUL( dBasisdx, TRANSPOSE( dBasisdx ) )
      DO q=1,nd
        ! Melt source
        ! ------------------------------
        FORCE(q) = FORCE(q) + Weight * LoadAtIP * Basis(q)

        !RHS FLUX TERM
        FORCE(q) = FORCE(q) + Weight * SUM( dBasisdx(q,1:dimsheet)*q0AtIP(1:dimsheet) )

        DO p=1,nd
          MASS(p,q) = MASS(p,q) + Weight * Basis(q) * Basis(p)
        !  IF((num_cold >0) .AND. (p .NE. q)) THEN 
        !    MASS(p,q) = 0._dp
        !  END IF
        END DO

      END DO
    END DO    


   num_cold = 0._dp
    IF (.NOT. CheckPassiveElement(Element) ) THEN
      DO i=1, n   ! ... for each node within the element (LOCAL nodal index)
        j = Element % NodeIndexes(i) 
        IF (ctvals(ctperm(j)) .LE. 0) THEN
          num_cold = num_cold + 1
        END IF
      END DO
      IF (num_cold > 0._dp) THEN
          !WRITE(*,*) "cold ACTIVE"
          STIFF = STIFF*0._dp
          FORCE = FORCE*0._dp
          MASS = MASS*0._dp
          DO i=1, n   ! ... for each node within the element (LOCAL nodal index)
            j = Element % NodeIndexes(i) 
            IF (ctvals(ctperm(j)) .LE. 0) THEN
            MASS(i,i) = 1
            END IF
          END DO
      END IF
    END IF
   
    
    IF(TransientSimulation) CALL Default1stOrderTime(MASS,STIFF,FORCE)
   


    !WRITE(*,*) "_______________________"

    CALL CondensateP( nd-nb, nb, STIFF, FORCE )
    CALL DefaultUpdateEquations(STIFF,FORCE)

!------------------------------------------------------------------------------
  END SUBROUTINE LocalMatrix
!------------------------------------------------------------------------------


! Assembly of the matrix entries arising from the Neumann and Robin conditions
!------------------------------------------------------------------------------
!  SUBROUTINE LocalMatrixBC( Element, n, nd )
!------------------------------------------------------------------------------
!    INTEGER :: n, nd
!    TYPE(Element_t), POINTER :: Element
!------------------------------------------------------------------------------
!    REAL(KIND=dp) :: Flux(n), Coeff(n), Ext_t(n), F,C,Ext, Weight
!    REAL(KIND=dp) :: Basis(nd),dBasisdx(nd,3),DetJ,LoadAtIP
!    REAL(KIND=dp) :: STIFF(nd,nd), FORCE(nd), LOAD(n)
!    LOGICAL :: Stat,Found
!    INTEGER :: i,t,p,q,dim
!    TYPE(GaussIntegrationPoints_t) :: IP

!    TYPE(ValueList_t), POINTER :: BC

!    TYPE(Nodes_t) :: Nodes
!    SAVE Nodes
!------------------------------------------------------------------------------
!    BC => GetBC()
!    IF (.NOT.ASSOCIATED(BC) ) RETURN

!    dim = CoordinateSystemDimension()

!    CALL GetElementNodes( Nodes )
!    STIFF = 0._dp
!    FORCE = 0._dp
!    LOAD = 0._dp

!    Flux(1:n)  = GetReal( BC,'field flux', Found )
!    Coeff(1:n) = GetReal( BC,'robin coefficient', Found )
!    Ext_t(1:n) = GetReal( BC,'external field', Found )

        
    ! Numerical integration:
    !-----------------------
!    IP = GaussPoints( Element )
!    DO t=1,IP % n
      ! Basis function values & derivatives at the integration point:
      !--------------------------------------------------------------
!      stat = ElementInfo( Element, Nodes, IP % U(t), IP % V(t), &
!              IP % W(t), detJ, Basis, dBasisdx )

!      Weight = IP % s(t) * DetJ

      ! Evaluate terms at the integration point:
      !------------------------------------------

      ! Given flux:
      ! -----------
!      F = SUM(Basis(1:n)*flux(1:n))

      ! Robin condition (C*(u-u_0)):
      ! ---------------------------
!      C = SUM(Basis(1:n)*coeff(1:n))
!      Ext = SUM(Basis(1:n)*ext_t(1:n))

!      DO p=1,nd
!        DO q=1,nd
!          STIFF(p,q) = STIFF(p,q) + Weight * C * Basis(q) * Basis(p)
!        END DO
!      END DO

!      FORCE(1:nd) = FORCE(1:nd) + Weight * (F + C*Ext) * Basis(1:nd)
!      TotLen = TotLen + Weight 
!    END DO
!    CALL DefaultUpdateEquations(STIFF,FORCE)
!------------------------------------------------------------------------------
!  END SUBROUTINE LocalMatrixBC
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
END SUBROUTINE SheetSolverhw
!------------------------------------------------------------------------------
