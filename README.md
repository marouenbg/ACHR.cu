[![DOI](http://joss.theoj.org/papers/10.21105/joss.01363/status.svg)](https://doi.org/10.21105/joss.01363)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3233085.svg)](https://doi.org/10.5281/zenodo.3233085)
[![CI](https://github.com/marouenbg/ACHR.cu/actions/workflows/vfwarmup.yml/badge.svg)](https://github.com/marouenbg/ACHR.cu/actions/workflows/vfwarmup.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/marouenbg/ACHR.cu/blob/master/LICENSE.txt)

# Description
ACHR.cu is a General Purpose Graphical Processing Unit (GP-GPU) implementation of the popular sampling algorithm of metabolic models ACHR. 

The code comes with the peer-reviewed software note [ACHR.cu: GPU accelerated sampling of metabolic networks.](http://joss.theoj.org/papers/10.21105/joss.01363).

Sampling is the tool of choice in metabolic modeling in unbiased analysis as it allows to explore the solution space constrained by the linear bounds without necessarily
assuming an objective function. As metabolic models grow in size to represent communities of bacteria and complex human tissues, sampling became less used because of the large analysis time.
With ACHR.cu we can achieve at least a speed up of 10x in the sampling process.

Technically sampling is a two step process:

1. The generation of warmup points.

2. The actual sampling starting from the previously generated warmup points.

# Installation
The software can be installed via cloning this repository to your local machine and compiling `VFWarmup` (for step 1) and `ACHRcu` (for step 2) at their root folders.
More details can be found in the [documentation](https://achrcu.readthedocs.io/en/latest/).

## Dependencies 

### VFWarmup (warmup point generation)

+ **LP solver** — one of:
  + [GLPK](https://www.gnu.org/software/glpk/) (open-source, recommended)
  + IBM CPLEX (free for academics)
+ MPI (OpenMPI or MPICH)
+ OpenMP (CPLEX version only — GLPK is not thread-safe)

### ACHRcu (GPU sampling)

+ CUDA >= 10.0
+ GSL (GNU Scientific Library)
+ IBM CPLEX

## Hardware requirements

+ NVIDIA GPU with sm_37 architecture (Kepler K80) or above
+ Check the [documentation](https://achrcu.readthedocs.io/en/latest/) for more details on the requirements.

# Reproducibility

Have a look at the [code ocean capusle](https://codeocean.com/capsule/2291048/tree/v1) to run ACHR.cu in an interactive container with a sample example. The capsule has an access to an 
NVIDIA GPU with all software
dependencies cached.

# Quick guide

Sampling is a two-step process:

## 1. Generation of warmup points

### Using GLPK (open-source)

```
cd VFWarmup
make glpk
```

Test your installation:
```
make test_glpk
```

Generate warmup points with MPI parallelism:
```
mpirun -np nCores ./createWarmupPtsGLPK model.mps nWarmupPoints
```

### Using CPLEX

```
cd VFWarmup
source ./install.sh
make
```

Make sure to run source on the install script because it exports environment variables. Then test your installation:
```
make test
```

Generate warmup points with hybrid MPI/OpenMP:
```
mpirun -np nCores --bind-to none -x OMP_NUM_THREADS=nThreads ./createWarmupPts model.mps nWarmupPoints
```

## 2. GPU sampling

Quick install:
```
cd ACHRcu
source ./install.sh
make
```
Also here, make sure to run source on the install script because it exports environment variables. Then, test your installation:
```
make test
```

Then you can perform the sampling:

```
./ACHRCuda model.mps warmuppoints.csv nFiles nPoints nSteps
```

This command generates the actual sampling points starting from the previously generated warmup points stored in `warmuppoints.csv` to produce a total of `nFiles*nPoints` with each point
requiring `nSteps` to converge.

# Acknowledgments

The experiments were carried out using the [HPC facilities of the University of Luxembourg](http://hpc.uni.lu)

# License

The software is free and is licensed under the MIT license, see the file [LICENSE](<https://github.com/marouenbg/ACHR.cu/blob/master/LICENSE.txt>) for details.

# Feedback/Issues

Please report any issues to the [issues page](https://github.com/marouenbg/ACHR.cu/issues).

# Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CONDUCT.md).
By participating in this project you agree to abide by its terms.

