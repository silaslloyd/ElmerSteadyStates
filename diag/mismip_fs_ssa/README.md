Elmer/Ice's own FS-SSA coupling test (elmerice/Tests/MISMIP_FS-SSA from ElmerCSC/elmerfem, GPL), copied unchanged
to check that Elmer's passive elements and "Passive Target" boundaries work in a given build, serially and in
parallel (jobs/passive_tests.cmd runs it on 1 and 4 ranks; the parallel copy uses MUMPS instead of UMFPACK).
