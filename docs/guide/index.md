# Usage guide

Sampling of metabolic models is a two-step process.

## Generation of warmup points

The generation of warmup points is done through `VFWarmup` software. After installing the dependencies of `VFWarmup`, you can build the binaries at the root of 
`VFWarmup` using `make`.

Then call `VFWarmup` as follows:

`mpirun -np nCores --bind-to none -x OMP_NUM_THREADS=nThreads createWarmupPts model.mps SCAIND`

Replace the following variables with your own parameters:

nCores: the number of non-shared memory cores you wish to use for teh analysis

nThreads: the number of shared memory threads within one core

model.mps: the metabolic model in `.mps` format. To convert a model in `.mat` format to `.mps`, you can use the provided converter `convertProblem.m`

SCAIND: (optional) corresponds to the scaling CPLEX parameter SCAIND and can take the values 0 (equilibration scaling: default), 1(aggressive scaling), -1 (no scaling).
scaling is usually desactivated with tightly constrained metabolic model such as coupled models to avoid numerical instabilities and large solution times.

Example: `mpirun -np 2 --bind-to none -x OMP_NUM_THREADS=4 createWarmupPts ecoli_core.mps`

You will have to input the number of warmup points to be generated, this is usually a minimum of 2*n, where n is the number of reactions in a metabolic model. `VFWarmup` will perform
a minimization and a maximization in each dimension, which means that 2*n is the minimum number of samples needed to delienate the solution space.

The ouput file is saved as `modelnPtswarmup.csv`, with model is the name of the metabolic model and nPts is the number of warmup points generated.

## Sampling

The actual GPU sampling is done through `ACHR.cu` software. After installing the dependencies of `ACHR.cu`, you can build the binaries at the root of `ACHR.cu` using `make`.

Then call `ACHR.cu` as follows:

`./ACHR model.mps warmuppoints.csv nFiles nPoints nSteps`

Replace the following varaibles with you own parameters:

model.mps: the metabolic model in `.mps` format.

warmuppoints.csv: the warmup points obtained using `VFWarmup`. You can also use warmup points generated using other software such as the MATLAB CobraToolbox and Cobrapy.

nFiles: Number of files to stores the sampled solution points.

nPoints: number of points per file.

nSteps: number of steps per point.

Example: `./ACHR ecoli_core.mps ecoli_core1000warmup.csv 1 1000 1000`


