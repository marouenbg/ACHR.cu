# Usage guide

Sampling of metabolic models is a two-step process.

## Generation of warmup points

The generation of warmup points is done through `VFWarmup` software. After installing the dependencies of `VFWarmup`, you can build the binaries at the root of 
`VFWarmup` using `make` (CPLEX) or `make glpk` (GLPK).

### GLPK version

```bash
mpirun -np nCores --bind-to none ./createWarmupPtsGLPK model.mps nWarmupPoints
```

Replace the following variables with your own parameters:

+ **nCores**: the number of MPI processes
+ **model.mps**: the metabolic model in `.mps` format. To convert a model in `.mat` format to `.mps`, you can use the provided converter `convertProblem.m`
+ **nWarmupPoints**: the number of warmup points to generate. This should be at least `2n`, where `n` is the number of reactions. If fewer points are requested, VFWarmup will automatically increase to `2n`.

Example:

```bash
mpirun -np 4 --bind-to none ./createWarmupPtsGLPK ecoli_core.mps 200
```

### CPLEX version

```bash
mpirun -np nCores --bind-to none -x OMP_NUM_THREADS=nThreads ./createWarmupPts model.mps nWarmupPoints
```

Additional parameter:

+ **nThreads**: the number of OpenMP threads within each MPI process. The CPLEX version uses hybrid MPI/OpenMP parallelism.
+ **SCAIND** (optional, appended after nWarmupPoints): the CPLEX scaling parameter. Values: `0` (equilibration scaling, default), `1` (aggressive scaling), `-1` (no scaling). Disable scaling for tightly constrained models (e.g., coupled models) to avoid numerical instabilities.

Example:

```bash
mpirun -np 2 --bind-to none -x OMP_NUM_THREADS=4 ./createWarmupPts ecoli_core.mps 200
```

### Output

The output file is saved as `modelnPtswarmup.csv`, where `model` is the name of the metabolic model and `nPts` is the actual number of warmup points generated.

## Sampling

The actual GPU sampling is done through `ACHRCuda`. After installing the dependencies of `ACHRCuda`, you can build the binary at the root of `ACHRcu` using `make`.

Then call `ACHRCuda` as follows:

```bash
./ACHRCuda model.mps warmuppoints.csv nFiles nPoints nSteps
```

Replace the following variables with your own parameters:

+ **model.mps**: the metabolic model in `.mps` format.
+ **warmuppoints.csv**: the warmup points obtained using `VFWarmup`. You can also use warmup points generated using other software such as the MATLAB CobraToolbox or COBRApy.
+ **nFiles**: number of output files to store the sampled solution points.
+ **nPoints**: number of points per file.
+ **nSteps**: number of steps per point (controls convergence).

Example:

```bash
./ACHRCuda ecoli_core.mps ecoli_core200warmup.csv 1 1000 1000
```

The output is written to files named `File0`, `File1`, etc. Each file contains `nReactions` rows and `nPoints` columns.


