# Tutorials

First, make sure that `VFWarmup` and `ACHR.cu` are correctly installed.

## Sampling of Ecoli core

Sampling is a two-step process. First, let's generate the warmup points for sampling using `VFWarmup`

`mpirun -np 1 --bind-to none -x OMP_NUM_THREADS=4 createWarmupPts ecoli_core.mps`

Make sur you pick a number of warmup points greater than 95*2=190, say 200 for instance. The output has been written to ecoli_core200warmup.csv

Then, let's sample the solution space of Ecoli core metabolic model starting from the warmup points generated previously:

./ACHR model.mps ecoli_core200warmup.csv 2 1000 1000

We will generate 2 files contatining each 1000 points. Each points has converged after 1000 step. The total number of points generated is 2*1000=2000.

