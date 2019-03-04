---
title: 'ACHR.cu: GPU sampling of metabolic networks.'
tags:
  - cuda
  - metabolism
  - constraint-based modeling
  - GPU
authors:
 - name: Marouen Ben Guebila
   orcid: 0000-0001-5934-966X
   affiliation: "1"
affiliations:
 - name: Luxembourg Centre for Systems Biomedicine, University of Luxembourg.
   index: 1
date: 03 March 2019
bibliography: paper.bib
---

# Summary

The *in silico* modeling of biological organisms consists of the mathematical representation of key functions of a biological system and the study of its behavior in different 
conditions and environments as a tool for the support of wet-lab experiment or to generate hypotheses about the functioning of its subsystems. Among the many biological layers, 
metabolism is the most amenable to modeling because it is directly related to key biological functions and is the support for several drugs targets, in addition to the largely 
available public data resources that document several metabolites and their abundances. As a biotechnological application, the metabolic modeling of ethanol-producing bacteria allows 
finding
key interventions (such as substrate optimization) that allow increasing the yield in the bioreactor, thereby its efficiency.
 
Recently, high-throughput technologies allowed to generate a large amount of biological data that enabled more complex modeling of biological systems. As models grew in size, the 
tools used for their analysis have to be appropriately scaled to include the use of parallel software.

A tool of choice for the analysis of metabolic models is the sampling of the space of their possible phenotypes. Instead of considering one specific biological function of interest, 
sampling is an unbiased tool for metabolic modeling. As models grow in size, sampling became expensive both in time and computational resources. To make sampling more accessible in the 
modeler ´s toolbox, I suggest ACHR.cu which is a CUDA based implementation of the sampling algorithm ACHR.

# Results

Mathematically, metabolic models are networks of metabolites involved in reactions and are formulated as linear programs.  The solution of the linear program allows finding flux 
distributions in the network that achieve the objective function of interest. ACHR allows obtaining flux distribution for each reaction under the conditions of optimality.

The sampling of the solution space of metabolic models is a two-step process:

## Generation of warmup points

The first step of sampling the solution space of metabolic networks involves the generation of warmup points that are solutions of the metabolic model's linear program. The sampling 
MCMC chain starts from those solutions to explore the solution space. The generation of p warmup points corresponds to flux variability analysis (FVA) solutions for the first p < 2n 
points, with n the number of reactions in the network, and corresponds to randomly generated solutions generated through a random objective vector c for the p > 2n points.

The generation of warmup points is a time-consuming process and requires the use of more than one core in parallel. The distribution of the points to generate among the c cores of the 
computer is often performed through static balancing; which means that each core gets p/c points to generate. Nevertheless, the formulation of the problem induces a significant 
imbalance in the distribution of work, meaning that the workers will not converge at the same time which slows down the overall process. I showed previously that FVA is imbalanced 
especially with metabolism-expression models. In this case, the generation of warmup points through random c vectors of objective coefficients is yet another factor to favor the 
imbalance between the points to generate.

To remediate to this issue, dynamic loading balancing implemented through the OpenMP parallel library in C allows assigning fewer points to workers that required more time to solve 
previous chunks. In the end, the workers converge at the same time.
Below, I report the run times of the generation of warmup points in MATLAB (CreateWarmupMATLAB) and through a hybrid MPI/OpenMP implementation (CreateWarmupVF), both for the generation 
of 30,000 warmup points. 
Since the original implementation in MATLAB does not support parallelism, I reported the run times for the sequential version below. The run times can be divided by the number of cores 
to get the times (at best) for a parallel run.
The experiments are the average of 3 trials in every setting in seconds.

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

The speedup is impressive (up to 50x in some cases) and shows the power of dynamic load balancing in imbalanced metabolic models.
Also, I noted that the model can be largely imbalanced due to the generation of a random c vector and that averaging three experiments can be insufficient to get the average run time 
and smooth out the outliers. In particular, run times between 16 and 32 cores were similar. Averaging more than three experiments can further show speed up between the settings.

## The actual sampling of the solution space starting from the warmup points.

The sampling of the solution space of metabolic models involves the generation of MCMC chains starting from the warmup points.
The sampling in MATLAB was performed using the ACHR serial function using one sampling chain, and the data was saved every 1000 points. The GPU parallel version creates one chain for 
each point. 
Each thread in the GPU executes one chain. Moreover, each thread can call additional threads to perform large matrix operations using the nested parallelism abilities of the new NVIDIA 
cards.   
In this case, the speed up with the GPU is quite important in the table below. It is noteworthy that even for a single core, the CPU is multithreaded especially with MATLAB base 
functions such as min and max.


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
 

*SVD and QR refer to the implementation of the null space computation.
 
The implementation of null space was a significant determinant in the final run time, and the fastest implementation was reported in the final run times.

While computing the SVD of the S matrix is more precise than QR, it is not prone to parallel computation in the GPU which can be even slower than the CPU in some cases.

Computing the null space through QR decomposition is faster but less precise and consumes more memory as it takes all the dimensions of the matrix as opposed to SVD that removes 
columns 
below a given precision of the SV.

# Comparison to existing software

The parallel GPU implementation of ACHR.cu is very similar to the MATLAB Cobra Toolbox 
[GpSampler](https://github.com/marouenbg/cobratoolbox/blob/master/src/modelAnalysis/sampling/ACHRSampler.m). 
Another tool, [OptGpSampler](http://cs.ru.nl/~wmegchel/optGpSampler/) provides a 40x speedup over GpSampler through a C implementation and fewer but longer sampling chains launch.
Since OptGpSampler performs the generation of the warmup points and the sampling in one process, it is clear from the results of this work that the speedup achieved with the generation 
of warmup points is more significant than sampling itself. I decoupled the generation of warmup points from sampling to take advantage of dynamic load balancing with OpenMP. In 
OptGpSampler, 
each worker gets the same amount of points and steps to compute; the problem is statically loaded by design.
While if we perform the generation of warmup points separately from sampling, the problem can be dynamically balanced because the workers can generate an uneven number of points. 

Finally, future improvements of this work can consider an MPI/CUDA hybrid to take advantage of the multi-GPU architecture of recent NVIDIA cards like the K80. Taken together, the 
parallel architecture of ACHR.cu allows faster processing of metabolic models thereby accelerating biomedical discovery.

# Acknowledgments

The experiments were carried out using the [HPC facilities of the University of Luxembourg](http://hpc.uni.lu)

# References
