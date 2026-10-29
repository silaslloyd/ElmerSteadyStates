Patches applied by jobs/build_elmer.cmd to Elmer's source (revision dd6aef5e2, elmerfem/2026-10-28) before building.

01-hypre-options.patch (fem/src/SolveHypre.c, fem/src/SParIterSolver.F90): Elmer's interface to Hypre
  1. BoomerAMG as a SOLVER is set up with the system matrix it solves with. With a separate preconditioning
     matrix (block preconditioning of a multi-component block, e.g. the Stokes velocity) it was set up with that
     matrix and then solved with the other, and crashed ("double free or corruption").
  2. BoomerAMG options for systems of PDEs and for the coarse levels, as new keywords (0 = Hypre's default):
       BoomerAMG Nodal = Integer n          HYPRE_BoomerAMGSetNodal (nodal coarsening: all components of a node
                                            together; 1 Frobenius, 4 row sum, 6 ... see the Hypre manual)
       BoomerAMG Nodal Diag = Integer n     HYPRE_BoomerAMGSetNodalDiag
       BoomerAMG Agg Num Levels = Integer n HYPRE_BoomerAMGSetAggNumLevels (aggressive coarsening, less memory)
       BoomerAMG P Max Elmts = Integer n    HYPRE_BoomerAMGSetPMaxElmts (interpolation entries per row)
       BoomerAMG Trunc Factor = Real r      HYPRE_BoomerAMGSetTruncFactor
  3. Hypre ILU: the options Elmer already reads are passed on (they were in an #if 0 block), plus the fill level:
       ILU Fill Level = Integer k (ILU(k) types), ILU Reordering, ILU Max Nonzeros Per Row, ILU Schur Max
       Iterations, ILU Drop Threshold (ILUT types)
Without the new keywords Elmer behaves as unpatched.
