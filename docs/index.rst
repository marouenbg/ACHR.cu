.. ACHRcu documentation master file, created by
   sphinx-quickstart on Sat Nov 24 10:24:49 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
==================================
ACHR.cu is a CUDA implementation of the sampling algorithm Artificially Centered Hit-and-Run (ACHR) for the analysis of metabolic models. Metabolic models are mathematical representations
of biological organisms forumlated as linear programs. Popular metabolic modeling tools like Flux Balance Analysis (FBA) assume an objective function that the organism optimizes for. 
When it is not obvious which objective function the system optimizes for, unbiased analysis like sampling is a tool of choice. Sampling is an MCMC method that explores the solution space or 
the set of possible phenotypes under the linear constraints. 

But as metabolic models of biological systems become more complex, the sampling of the solution space of a metabolic model becomes unfeasible due to the large analysis time. 
In order to address the large analysis time for large metabolic models, I implemented a GP-GPU version of ACHR that reduces the sampling time by at least a factor of 10x for the sampling per se
and a factor of 100x for the generation of warmup points which is the preprocessing step.
Here you can find tutorials on the installation and analysis of ACHR.cu sampling software.

General approach and parallel construct
========================================
Sampling metabolic models is a two-step process:

1. Generation of warmup points.
The generation of p warmup points is basically solving the linear program with randomly generated coefficient vector `c` twice as a maximization problem and a minimization problem.
The use of a randomly genrated coefficient vector makes the solution of linear program extremely slow and subject to numerical instability. Particularly in parallel setting, some
cores might get the linear programs that require more time to solve while others get the easier ones, which can result in an overall slower analysis time. In a previous work, I addressed a similar question through dynamic load balancing. 
Briefly, if a worker gets a high computational load then the idle workers can take up some of that load. Using a dynamically load balanced generation of warmup points software, the speed up achieved is 
at least a 100x.

2. The actual sampling using the warmup points as a starting point.
With the warmup points at hand, we can proceed to the actual sampling using a cuda implementation. The architecture uses the modern specs of Nvidia cards to perform [dynamic parallelism](http://developer.download.nvidia.com/assets/cuda/files/CUDADownloads/TechBrief_Dynamic_Parallelism_in_CUDA.pdf).
In fact, there will be p random starting points at the same time (first level of parallelism) that will each launch n random chains to sample the solution space. This procedure is repeated
a number of times taking each time a new starting point and saving the sampled points.
In particular each chain will sample the local space close to its starting point, which could improve the convergence of the algorithm and avoid the blocking of the sampling chain in 
the corners of the flux cone. Additionally, the provided computational power will allow the user to sample a greater number of points which can greatly help the assessement of the uniform
representation of the solution space and address the sampling of large metabolic models.

Contents
========

.. toctree::
   :hidden:

   self

.. toctree::

   install/index

.. toctree::

   guide/index

.. toctree::

   tutos/index

.. toctree::

   changelog

.. toctree::

   license/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
