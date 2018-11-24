.. ACHRcu documentation master file, created by
   sphinx-quickstart on Sat Nov 24 10:24:49 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
==================================
ACHR.cu is a CUDA implementation of the sampling algorithm Artificially centered Hit-and-Run(ACHR) for the analysis of metabolic models. As metabolic models of biological systems become
more complex, the sampling of the solution space of a metabolic model becomes unfeasible due to the large analysis time. Although, sampling has interesting advantages as it is unbiased to any
objective function. In order to address the large analysis time for large metabolic models, I implemented a GP-GPU version of ACHR that reduces the sampling time but at least a factor of 10x.
Here you can find tutorials on the installation and analysis of ACHR.cu samplng software.

General approach
==================
Sampling metabolic models is a two-step process:

1. Generation of warmup points.

2. The actual sampling using the warmup points as a starting point.

.. toctree::
   :hidden:

   self

.. toctree::

   install/index

.. toctree::

   guide/index

.. toctree::

   license/index

.. toctree::

   tutos/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
