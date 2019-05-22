---
title: 'ACHR.cu: GPU accelerated sampling of metabolic networks.'
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

# Introduction

The *in silico* modeling of biological organisms consists of the mathematical representation of key functions of a biological system and the study of its behavior in different conditions and environments. It serves as a tool for the support of wet lab experiments and for the generation of hypotheses about the functioning of the subsystems. Among the many biological 
products, 
metabolism is the most amenable to modeling because it is directly related to key biological functions and is the support for several drugs targets. 
Moreover, public data resources of several metabolites and their abundances have been developing rapidly in recent years thereby enabling applications in many areas. In biotechnology, the metabolic modeling 
of ethanol-producing bacteria allows 
finding key interventions (such as substrate optimization) that would increase the yield in the bioreactor, thereby its efficiency [@mahadevan2005applications].
 
Recently, high-throughput technologies allowed to generate a large amount of biological data that enabled complex modeling of biological systems. As models expand in size, the 
tools used for their analysis have to be appropriately scaled to include the use of parallel software.

A tool of choice for the analysis of metabolic models is the sampling of the space of their possible phenotypes. Instead of considering one specific biological function of interest, 
sampling is an unbiased tool for metabolic modeling that explores all the space of possible metabolic phenotypes. For large models, sampling becomes expensive both in time and computational resources. To make 
sampling accessible in the 
modeler´s toolbox, I present ACHR.cu which is a fast, CUDA-based [@nickolls2008scalable] implementation of the sampling algorithm ACHR [@kaufman1998direction].

# Results

Metabolic models are networks of m metabolites involved in n reactions. Briefly, they are formulated as linear programs [@o2015using] using the stoichiometric matrix S(m,n). A central hypothesis in metabolic modeling is the steady state assumption, meaning that generating fluxes are equal to degrading fluxes and the rate of change of metabolite concentrations is null. Mathematically, the steady state assumption translates to constraining the solution space to the null space of the stoichiometric matrix S(m,n). 

Particularly, the solution of the linear program allows finding flux distributions in the network that achieve the objective function of interest. ACHR allows obtaining flux distribution for each reaction.

The sampling of the solution space of metabolic models is a two-step process:

## Generation of warmup points

The first step of sampling the solution space of metabolic models involves the generation of warmup points that are solutions to the metabolic model's linear program. The sampling 
Markov Chain Monte Carlo (MCMC) chain starts from those solutions to explore the solution space. 

The generation of p <a href="https://www.codecogs.com/eqnedit.php?latex=\geq" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\geq" title="\geq" /></a> 2n warmup points corresponds to flux variability analysis (FVA) [@mahadevan2003effects] solutions 
for the first 2n points, with n the number of reactions in the network and the objective function is to minimize and maximize each reaction in the model (hence 2n). For the remaining p - 2n points, it corresponds to solutions generated using a random objective vector c(n,1).

The generation of warmup points is a time-consuming process and requires the use of more than one core in parallel. The distribution of the points to generate among the nc cores of the computer is often performed through static balancing with each core getting p/nc points to generate. Nevertheless, the formulation of the problem induces a significant imbalance in the distribution of work, meaning that the workers will not converge at the same time thereby slowing down the overall process. I showed previously that FVA is imbalanced, 
especially with metabolism-expression models [@guebila2018dynamic]. The generation of warmup points through random c vectors of objective coefficients is yet another factor to favor ill-conditioned problems and the imbalance between the parallel workers.

To address the imbalance between the workers, dynamic loading balancing as implemented through the OpenMP parallel library in C [@dagum1998openmp] allows assigning fewer points to workers that required more time to solve previous chunks of reactions. In the end, the workers converge at the same time.

Given this background, the speedup of the generation of warmup points using the hybrid MPI/OpenMP implementation (CreateWarmupVF) over the MATLAB version (CreateWarmupMATLAB) was substantial 
(up to 50x in some cases) [@guebila2018dynamic] and showed the power of dynamic load balancing in imbalanced metabolic models. Using the generated warmup points, the uniform sampling process can start to explore the solution space.

## Sampling of the solution space

Second, the sampling of the solution space of metabolic models involves the generation of MCMC chains starting from the warmup points.
The sampling in MATLAB was performed using the ACHR sequential function using one sampling chain, and the data was saved every 1000 points. The GPU parallel version (ACHR.cu) creates one chain for each point executed by one thread in the GPU. Moreover, each thread can call additional threads to perform large matrix operations using the grid nesting and dynamic parallelism capabilities of the new NVIDIA cards (sm_35 and higher).   
When compared to the CPU, the speedup with the GPU is quite important as reported in table 1. It is noteworthy that even for a single core, the CPU is multithreaded especially with optimized MATLAB 
base functions such as min and max.


| Model         | Points             | Steps           |Intel Xeon (3.5 Ghz)  |Tesla K40    |
| --------------| -------------------| ----------------|----------------------|-------------|
| Ecoli core    | 1000               | 1000            |42                    | 2.9   (SVD) |      
| Ecoli core    | 5000               | 1000            |208                   | 12.5  (SVD) |
| Ecoli core    | 10000              | 1000            |420                   | 24.26 (SVD) |
| P Putida      | 1000               | 1000            |103                   | 17.5  (SVD) |
| P Putida      | 5000               | 1000            |516                   | 70.84 (SVD) |
| P Putida      | 10000              | 1000            |1081                  | 138   (SVD) |
| Recon2        | 1000               | 1000            |2815                  | 269   (QR)  |
| Recon2        | 5000               | 1000            |14014                 | 1110  (QR)  |
| Recon2        | 10000              | 1000            |28026                 | 2240  (QR)  |
 
Table 1: Runtimes of ACHR in MATLAB and ACHR.cu for a set of metabolic models starting from 30,000 warmup points. *SVD and QR refer to the implementation of the null space computation.

The implementation of null space computation to constrain the metabolic model was a significant determinant in the final runtime, and the fastest implementation was reported in the final result (Table 1). Particularly, there was a tradeoff in memory usage and access as opposed to computation time when either QR or Singular Value Decomposition (SVD) were used.

While computing the SVD of the S matrix is generally more precise than QR, it is not prone to parallel computation in the GPU which can be even slower than the CPU in some cases. However, computing the null space through QR decomposition is faster but less precise and consumes more memory as it takes all the dimensions of the matrix in contrast to SVD that removes 
columns below a given precision of the singular values.

Finally, ACHR.cu was developed as a high-performance tool for the modeling of metabolic networks using a parallel architecture that segregates the generation of warmup points and the sampling.

# Comparison to existing software

Conceptually, the architecture of the parallel GPU implementation of ACHR.cu is similar to the MATLAB Cobra Toolbox [@heirendt2019creation] GpSampler. 
Another tool, OptGpSampler [@megchelenbrink2014optgpsampler] provides a 40x speedup over GpSampler through a i) C implementation and ii) fewer but longer sampling chains launch.
Since OptGpSampler performs the generation of the warmup points and the sampling in one process, it is clear from the results of the current work that the speedup achieved with the generation of warmup points is more significant than sampling itself. I decoupled the generation of warmup points from sampling to take advantage of dynamic load balancing with OpenMP. Additionally, in OptGpSampler each worker gets the same amount of points and steps to compute; the problem is statically load balanced by design.
In contrast, if the generation of warmup points is performed separately from sampling, the problem can be dynamically balanced because the workers can generate an uneven number of warmup points and converge simulatenously. 

Finally, future improvements of this work can consider an MPI/CUDA hybrid to take advantage of the multi-GPU architecture of recent NVIDIA cards like the K80. Taken together, the 
parallel architecture of ACHR.cu allows faster sampling of metabolic models over existing tools thereby enabling the unbiased analyses of large-scale systems biology models.

# Acknowledgments

The experiments presented in this paper were carried out using the HPC facilities of the University of Luxembourg [@VBCG_HPCS14] -- see [https://hpc.uni.lu](https://hpc.uni.lu).

# References
