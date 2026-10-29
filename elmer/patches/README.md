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

02-column-coarse.patch (fem/src/SParIterSolver.F90, fem/src/IterSolve.F90): a two-level preconditioner for
extruded meshes, for Elmer's own parallel Krylov solvers (GCR, BiCGStab, ... with Linear System Solver = Iterative)
  Partition-local ILU (block Jacobi) cannot see modes that span many partitions; on sliding or floating ice the
  vertically uniform (plug-flow) modes are nearly free and stall ILU0 on compact partitions. The patch adds a coarse
  level of those modes: per mesh column (the nodes above one node of the 2D footprint) a uniform horizontal velocity
  (u, v) and a uniform pressure. Its Galerkin operator C = P^T A P (about the SSA operator of the current
  linearisation) is gathered to rank 0 and factorised by MUMPS once per linear solve; every application is
      x1 = P C^-1 P^T r,   z = x1 + M^-1 (r - A x1)      (M: the Linear System Preconditioning, ILUn/ILUT/diagonal/none)
  Columns from the node numbering of Elmer's internal extrusion (checked on every node; a Fatal error otherwise);
  in the normal-tangential frame the modes are rotated per node; Dirichlet dofs carry no coarse component.
       Linear System Column Coarse = Logical True
       Linear System Column Coarse Pressure = Logical     (default True; False: velocity modes only)
       Linear System Column Coarse Levels = Integer       (node layers; default the Simulation's Extruded Mesh Levels)
       Linear System Column Coarse Ranks = Integer        (ranks of the MUMPS coarse solve; default 1)
  The log reports per solve the coarse size, setup time and the time in the coarse corrections ("ColumnCoarse:").
  Hook: IterSolve's public IterSolverPrecWrap (an address) replaces the preconditioner IterSolver has just set up,
  once. Without the keyword Elmer behaves as unpatched. Not compiled outside tethys yet: the first build checks it.
