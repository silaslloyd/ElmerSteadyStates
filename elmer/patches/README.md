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
  level of those modes: per mesh column (the nodes above one node of the 2D footprint) u, v and w, each uniform and
  linear in the relative height (Order 1, the default; Order 0: uniform u, v only, which made the full residual 8x
  larger at each coarse step: w and incompressibility were left out), and a uniform pressure. Its Galerkin operator C = P^T A P (about the SSA operator of the current
  linearisation) is gathered to rank 0 and factorised by MUMPS once per linear solve; every application is
      x1 = P C^-1 P^T r,   z = x1 + M^-1 (r - A x1)      (M: the Linear System Preconditioning, ILUn/ILUT/diagonal/none)
  Columns from the node numbering of Elmer's internal extrusion (checked on every node; a Fatal error otherwise);
  in the normal-tangential frame the modes are rotated per node; Dirichlet dofs carry no coarse component.
       Linear System Column Coarse = Logical True
       Linear System Column Coarse Pressure = Logical     (default True; False: velocity modes only)
       Linear System Column Coarse Levels = Integer       (node layers; default the Simulation's Extruded Mesh Levels)
       Linear System Column Coarse Ranks = Integer        (ranks of the MUMPS coarse solve; default 1)
       Linear System Column Coarse Order = Integer        (degree in height of the velocity modes; default 1)
       Linear System Column Coarse Pre Smoothing = Logical (ILU before and after the coarse step; default False)
       Linear System Column Coarse Recompute = Integer    (rebuild C and its factors every n-th solve; default 1)
       Linear System Column Coarse Below Variable / Below Value, Above Variable / Above Value, Grow: coarse modes
         only in the columns where a variable is below / above a value (e.g. GroundedMask < 0.5: floating and
         grounding line; ColdTempMask > 0: temperate bed), plus Grow rings of neighbours; compact numbering
  The log reports per solve the coarse size, setup time and the time in the coarse corrections ("ColumnCoarse:").
  Hook: IterSolve's public IterSolverPrecWrap (an address) replaces the preconditioner IterSolver has just set up,
  once. Without the keyword Elmer behaves as unpatched.
  Every dof takes the coarse weights of its owning partition (sharing partitions may differ in Dirichlet flags or
  rotation), so C is exactly P^T A P. Diagnostics for the first 3 applications per solve ("Linear System Column
  Coarse Diagnostics" = n): |P^T (r - A x1)| / |P^T r| (Galerkin consistency, ~1e-12), |r - A x1|/|r|, and the
  sizes of the coarse and local corrections.

04-bfbt.patch (fem/src/BlockSolve.F90): the BFBt approximation of the Stokes Schur complement for the block
preconditioner's pressure block
  Block BFBt = Logical True: the LAST block (the pressure) solved by z = K^-1 (C - B A Bt) K^-1 r, K = C - B Bt, in
  Elmer's diagonally scaled system (diag of the velocity block = 1 there: BFBt with D = diag(A) becomes D = I; the
  stabilised form keeps the bubbles' pressure block C in both factors, exact for D = A). Every product with B, Bt, A
  and C is a part of the monolithic product (BlockMatrixVectorProdMono, correct in parallel).
  K^-1: a flexible CG on -K, preconditioned by -K = B Bt - C ASSEMBLED once per linear solve from the block matrices
  (partition-local products: exact in serial, approximate at the partition interfaces in parallel; parallel
  structures from the block's variable, as BlockSolveInt makes them) and solved with the "block NN:" keywords (one
  BoomerAMG V-cycle). (The first version preconditioned the K solves by C alone: CG needed ~1700 iterations to 1e-2 on
  the lava matrices, the K solves stopped at 50 unsolved and the outer GCR stalled at ~0.96.)
       Block BFBt Tolerance = Real (the K solves' relative residual; default 1e-2)
       Block BFBt Max Iterations = Integer (default 50)
       Block BFBt Output = Integer (every n-th CG iteration's residual; default 0: none)
       Block BFBt Assembled K = Logical (the K solves' CG products with the assembled -K, one pressure-sized
         product instead of two monolithic ones; exact in serial, approximate at partition interfaces; default False)
  The pressure block must be solved last (natural Block Order). Without the keyword Elmer behaves as unpatched.
  Use: elmer/stokes_block_bfbt.sif.in. Measured on Elmer's own matrices (the tiny lava case, exact velocity solves),
  outer GCR iterations, the 1/mu pressure mass -> BFBt: viscosity jump x 1e4 270 -> 67, uniform 74 -> 45, no-slip
  walls 198 -> 78.

05-bfbt-hypre-k.patch (fem/src/BlockSolve.F90 after 04, fem/src/SolveHypre.c after 01): BFBt's K exact in parallel
  Block BFBt Hypre K = Logical True: -K = sum_c B_c Bt_c - C formed in HYPRE from the globally assembled blocks, and
  the K solves by Hypre's PCG (Block BFBt Tolerance / Max Iterations) with one BoomerAMG V-cycle as preconditioner
  (the "block NN:" BoomerAMG keywords of the pressure block: Relax Type (default 6, symmetric: PCG), Coarsen Type
  (10), Interpolation Type (6), Agg Num Levels, P Max Elmts, Max Levels, Num Sweeps, Strong Threshold (0.25)).
  Why: Elmer's local block matrices are unassembled at the partition interfaces (the rows of shared dofs hold partial
  sums; SolveHypre1 adds all local rows and Hypre sums them), and patch 04's partition-local products
  sum_r B_r Bt_r miss the cross terms B_r Bt_s there: K wrong next to every interface. Measured (the coarse sticky
  case, one step, outer GCR to 1e-6): 1 rank 45; assembled K 8 ranks 74, 64 ranks 84; matrix-free K 64 ranks 62.
  Here each block goes to a Hypre IJ matrix with Elmer's continuous global numbering (ContinuousNumbering, as
  SolveHypre), the products by hypre_ParCSRMatMat / hypre_ParCSRMatrixAdd (Hypre's internal API: _hypre_parcsr_mv.h),
  built once per linear solve. The packed (owned) pressure vectors are Hypre's local vectors (order checked; a Fatal
  error otherwise). The log: "BFBt: -K = B Bt - C in Hypre (n entries) ..." at Level 5; with Block BFBt Output > 0
  every K solve's PCG iterations and residual. Without the keyword patch 04 behaves as before.

06-bfbt-boundary.patch (after 05; BlockSolve.F90, SolveHypre.c): the boundary-adjusted BFBt (Elman & Tuminaro,
ETNA 35, 2009, the boundary-adjusted least-squares commutator)
  Block BFBt Boundary Weight = Real eps (> 0; the paper's 0.1; needs Block BFBt Hypre K):
    S~^-1 = K^-1 (C - B A W Bt) K_W^-1,   K = C - B Bt,   K_W = C - B W Bt
  W diagonal: 1, except eps at the velocity dofs of component c coupled by B to a pressure node on a boundary to
  which c is TANGENTIAL (all boundary elements; the normal's dominant axis; flags summed over the partitions). The
  plain BFBt's commutator is wrong in a layer along the boundaries and its iterations grow with refinement; measured
  (the fine sticky geometry, one step, Hypre K): 20x4 cm 34, 10x2 cm 80, 5x1 cm no convergence (0.19 after 80);
  Elmer's mass-matrix block 60, 70, ~38. K_W: a second Hypre matrix, PCG and BoomerAMG (same keywords). The log:
  "BFBt: boundary weight eps on n of N velocity dofs" (Level 4). Without the keyword 05 behaves as before.
