[![DOI](https://zenodo.org/badge/133329310.svg)](https://zenodo.org/badge/latestdoi/133329310)
[![TRAVIS](https://travis-ci.com/marouenbg/ACHR.cu.svg?branch=master)](https://travis-ci.com/marouenbg/ACHR.cu)
[![codecov](https://codecov.io/gh/marouenbg/ACHR.cu/branch/master/graph/badge.svg)](https://codecov.io/gh/marouenbg/ACHR.cu)

This repo contains the code for ACHR.cu the cuda GPU implementation of the sampling algorithm ACHR.

The repo also contains VFwarmup which is an implementation of the generation of sampling warmup points using dynamic load balancing.

ACHR is an MCMC sampling algorithm that is widely used in metabolic models. ACHR allows to obtain flux distribution for each reaction under the conditions of optimality.

The sampling of the solution space of metabolic models is a two-step process:

### Generation of warmup points

The first step of sampling the solution space of metabolic networks involves the generation of warmup points that are solutions of the metabolic model's linear program. The sampling MCMC chain starts from those solutions to explore the solution space. The generation of p warmup points corresponds to flux variability analysis (FVA) solutions for the first p < 2n points, with n the number of reactions in the network, and to randomly generated solutions genrated through a randomc objective vector c for the p > 2n points.

The genration of warmup points is a time-consuming process and requires the use of more than one core in parallel. The distribution of the points to generate among the c cores of the computer is often performed through static balancing; which means that each core gets p/c points to generate. Nevertheless, the formulation of the problem induces a large imbalance in the distribution of work, meaning that the workers will not converge in the same time which slows dowmn the overall process. I showed previously that FVA is imbalanced especially with metabolism-expression models. In this case, the generation of warmup points through random c vectors of objective coefficients is yet another factor to favor the imbalance between the points to generate.

To remediate to this issue, dynamic loading balancing implemented through the OpenMP parallel library in C allows to assign less points to workers that required more time to solve previous chunks. In the end, the workers converge at the same time.
Below I report the run times of the generation of warmup points in MATLAB (CreateWarmupMATLAB) and through a hybrid MPI/OpenMP impelementation (CreateWarmupVF), both for the generation of 30,000 warmup points. 
Since the original implementation in MATLAB does not support parallelism, I reported the run times for the sequential version below. We can divide by the number of cores to get the times (at best) for a parallel version.
The experiments are the average of 3 trials in every settings in seconds.

| Model         | CreateWarmupMATLAB | CreateWarmupVF  |CreateWarmupVF  |CreateWarmupVF |CreateWarmupVF |CreateWarmupVF |CreateWarmupVF |
| ------------- |:------------------:| ---------------:|----------------|---|-------|---|---|
| Cores         | 1                  | 1               |2               |4  |8      |16 |32 |
| Ecoli core    |149                 |2.8              |1.8             |0.8|0.7    |0.5|0.5|
| P Putida      | 385                | 12.5            |13              |8  |4      |2  |2  |
| EcoliK12      | 801                |    49           |43              |23 |10.4   |9.5|9.1|
| Recon2        | 11346              |     288         |186             |30 |32     |24 |21 |
| E Matrix      | NA*                |   602           |508             |130|52     |43 |43 |
| Ec Matrix     | NA*                | 5275            |4986            |924|224    |118|117|

*did not converge after 20,000 seconds.

The speed up is impressive (up to 50x in some cases) and shows the power of dynamic load balancing in imbalanced metabolic models.
Also, I noted that would the model can be largely imbalanced due to the generation of a random c vector, that averaging 3 experiments can be insufficient to get the mean run time and smooth out the outliers. In particular run times between 16 and 32 cores were similar. Averaging more than 3 experiments can further show speed up between the settings.

1. Software and hardware requirements

createWarmupVF requires OpenMP and MPI through OpenMPI implementation, and IBM CPLEX 12.6.3.

2. Quick guide

After successful make, the call to createWarmupVF is performed as follows:

`mpirun -np nCores --bind-to none -x OMP_NUM_THREADS=nThreads createWarmupPts model.mps SCAIND`

with nCores the number of devices that do not share memory, nThreads is the number of memory-sharing threads, SCAIND can be -1 ,+1, or 0 for the different scaling parameters of CPLEX. -1 is recommended for metabolism expression models.

### The actual sampling of the solution space starting from the warmup points.

Sampling of the solution space of metabolic models involves the generation of MCMC chains starting from the warmup points.
The sampling in MATLAB was performed using the ACHR serial function using one sampling chain and the data was saved every 1000 points. The GPU parallel version creates one chain for each point. 
Each thread in the GPU executes one chain. Morevoer, each thread can call additional threads to perform large matrix operations using the nested parallelism abilities of the new NVIDIA cards.   
In this case, the speed up with the GPU is quite important in the table below. It is noteworthy that even for a single core, the CPU is multithreaded especially with MATLAB base functions such as min and max.


| Model         | Points             | Steps           |Intel Xeon (3.5 Ghz)  |Tesla K40    |
| ------------- |:------------------:| ---------------:|----------------------|-------------|
| Ecoli core    | 1000               | 1000            |42                    | 2.9   (SVD) |      |
| Ecoli core    | 5000               | 1000            |208                   | 12.5  (SVD) |
| Ecoli core    | 10000              | 1000            |420                   | 24.26 (SVD) |
| P Putida      | 1000               | 1000            |103                   | 17.5  (SVD) |
| P Putida      | 5000               | 1000            |516                   | 70.84 (SVD) |
| P Putida      | 10000              | 1000            |1081                  | 138   (SVD) |
| Recon2        | 1000               | 1000            |2815                  | 269   (QR)  |
| Recon2        | 5000               | 1000            |14014                 | 1110  (QR)  |
| Recon2        | 10000              | 1000            |28026                 | 2240  (QR)  |
 

*SVD and QR refer to the impelementation of the null space computation.
 
The implementation of null space was a major determinant in the final run time and the fastest implementation was reported in the final run times.

While computing the SVD of the S matrix is more precise than QR, it is not prone for parallel computation in the GPU which can be even slower than the CPU in some cases.

Computing the null space through QR decompostion is faster but less precise and consumes more memory as it takes all the dimensions of the matrix as opposed to SVD that removes colmuns below a given precision of the SV.

1. Software and hardware requirements

ACHR.cu requires CUDA 8.0 and works on sm_35 hardware and above. This is particularly due to the use of nested parallelism that is only available in these versions.
It also requires CPLEX 12.6.3 and GSL library of linear algebra (for the sequential version of SVD and QR).

2. Quick guide

After successful make, the call to ACHR.cu is perfomred as follows:

`./ACHR model.mps warmuppoints.csv nFiles nPoints nSteps`

The model has to be provided in `.mps` format, the warmup points are the ouput file of the createWarmupVF, nFiles is the number of files in the output, nPoints is the number
of points per file, and nSteps is the number of steps per point.

### Comparison to existing software

The parallel GPU implementation of ACHR.cu is very similar to the MATLAB Cobra Toolbox [GpSampler](https://github.com/marouenbg/cobratoolbox/blob/master/src/modelAnalysis/sampling/ACHRSampler.m). 
[OptGpSampler](http://cs.ru.nl/~wmegchel/optGpSampler/) provides a 40x speedup over GpSampler through a C implementation and fewer but longer sampling chains launches.
Since OptGpSampler performs the generation of the warmup points and the sampling in one process, it is clear from the results of this work that the speedup achieved with the generation of warmup points is greater than sampling itself. I decoupled the generation of warmup points from sampling to take advantage of dynamic load balancing with OpenMP. In OptGpSampler, each worker gets the same amount of points and steps to compute, the problem is statsically loaded by design.
While if we perform the generation of warmup points separetly from sampling, the problem can be dynamically balanced because the workers can generate uneven number of points. 

### Future improvements

+ Potentially an MPI/CUDA hybrid to take advantage of the multi-GPU arhcitecture of recent NVIDIA cards like the K80.

+ Merge the warmup points generation and the sampling processes in one call.

### Acknowledgments

The experiments were carried out using the [HPC facilities of the University of Luxembourg](http://hpc.uni.lu)

# License

The project is licensed under the MIT license, see the file [LICENSE](<https://github.com/marouenbg/ACHR.cu/blob/master/LICENSE>) for details.
