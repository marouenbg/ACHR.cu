==========
Changelog
==========

0.3.0 (2026-03-31)
-------------------

Bug fixes
^^^^^^^^^

* Fix OpenMP data race in CPLEX FVA (shared ``x``/``values``/``objval`` buffers)
* Fix ``realloc`` on local pointer in null space computation (caller never sees new allocation)
* Fix double-free of ``d_Slin`` in ``computeKernelCuda``
* Fix ``rand()%1`` always returning 0 in random objective generation
* Fix ``MPI_Allreduce`` on non-contiguous ``double**`` memory
* Fix uninitialized ``solstat`` and ``nPts`` variables
* Fix buffer overflow in filename concatenation
* Fix ``sizeof(double*)`` instead of ``sizeof(double)`` in ``fluxMat`` allocation
* Fix memory leaks in ``movePtsBds``, ``fva``, and ``main``

New features
^^^^^^^^^^^^

* Add GLPK-based warmup point generator (``createWarmupPtsGLPK``) — no CPLEX license required
* Command-line arguments replace interactive ``scanf`` for ``nPts``
* GitHub Actions CI for VFWarmup (GLPK)

Infrastructure
^^^^^^^^^^^^^^

* Update CUDA arch to ``sm_37`` for K80 GPUs
* Replace deprecated ``__CUDACC_VER__`` with ``__CUDACC_VER_MAJOR__``
* Replace deprecated ``math_functions.h`` include

0.1.0 (2018-10-22)
-------------------

* Initial release
* Quick install script
* Documentation on Read the Docs
