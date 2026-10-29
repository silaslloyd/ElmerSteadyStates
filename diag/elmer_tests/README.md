Two of Elmer's own test cases (ElmerCSC/elmerfem fem/tests, GPL), copied unchanged, to check Elmer's multigrid
paths on an installation outside ESS (jobs/mg_tests.cmd): BlockLinElast3b (block solver, blocks by Elmer's own
cluster multigrid, `gcr_cmg_sgs.sif`) and mgdyn_airgap2 (block solver with Hypre BoomerAMG, Elmer's runs: 1, 2, 8
ranks). runtest.cmake shows how Elmer's test suite meshes them. ElmerSolver writes TEST.PASSED (1 = passed).
