==========
Changelog
==========

* :bug:`-` Fix OpenMP data race in CPLEX FVA (shared x/values/objval buffers)
* :bug:`-` Fix realloc on local pointer in null space computation (caller never sees new allocation)
* :bug:`-` Fix double-free of d_Slin in computeKernelCuda
* :bug:`-` Fix rand()%1 always returning 0 in random objective generation
* :bug:`-` Fix MPI_Allreduce on non-contiguous double** memory
* :bug:`-` Fix uninitialized solstat and nPts variables
* :bug:`-` Fix buffer overflow in filename concatenation
* :bug:`-` Fix sizeof(double*) instead of sizeof(double) in fluxMat allocation
* :bug:`-` Fix memory leaks in movePtsBds, fva, and main
* :feature:`-` Add GLPK-based warmup point generator (createWarmupPtsGLPK)
* :feature:`-` Command-line arguments replace interactive scanf for nPts
* :support:`-` Update CUDA arch to sm_37 for K80 GPUs
* :support:`-` Replace deprecated __CUDACC_VER__ with __CUDACC_VER_MAJOR__
* :support:`-` Replace deprecated math_functions.h include
* :release:`0.3.0 <2026.03.31>`

* :feature:`-` Added quick install script
* :support:`-` Improve the changelog structure
* :feature:`-` Changelog added to the doc
* :feature:`-` Improve the docs
* :release:`0.1.0 <2018.10.22>`
