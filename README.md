This repo contains the code for ACHR.cu the cuda GPU implementation of the sampling algorithm ACHR.

The repo also contains VFwarmup which an implementation of the generation of sampling warmup points using dynamic load balancing.

Sampling of the solution space of metabolic models is a two-step process:

1- Generation of warmup points

2- The actual sampling of the solution space starting from the warmup points.
