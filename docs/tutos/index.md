# Tutorials

First, make sure that `VFWarmup` and `ACHRCuda` are correctly installed (see [Installation guide](../install/index.md)).

## Sampling of P. putida

This tutorial demonstrates the full sampling workflow using the P. putida metabolic model (1060 reactions, 911 metabolites).

### Step 1: Generate warmup points

Using the GLPK version (no CPLEX license required):

```bash
cd VFWarmup
mpirun -np 4 --bind-to none ./createWarmupPtsGLPK lib/P_Putida.mps 5000
```

This generates `lib/P_Putida5000warmup.csv` with 1060 rows (reactions) and 5000 columns (warmup points). Note: if `nPts < 2 * nReactions`, VFWarmup automatically increases it to `2 * nReactions`.

Alternatively, using the CPLEX version:

```bash
mpirun -np 2 --bind-to none -x OMP_NUM_THREADS=4 ./createWarmupPts lib/P_Putida.mps 5000
```

### Step 2: Sample the solution space

```bash
cd ACHRcu
./ACHRCuda ../VFWarmup/lib/P_Putida.mps ../VFWarmup/lib/P_Putida5000warmup.csv 2 1000 1000
```

This generates 2 files (`File0`, `File1`) containing 1000 sampled points each, with 1000 steps per point for convergence. The total number of sampled points is `2 * 1000 = 2000`.

## Sampling of E. coli core

A smaller example using the E. coli core model (95 reactions).

### Step 1: Generate warmup points

```bash
cd VFWarmup
mpirun -np 1 --bind-to none ./createWarmupPtsGLPK ecoli_core.mps 200
```

Since `95 * 2 = 190 < 200`, VFWarmup will generate exactly 200 warmup points.

### Step 2: Sample

```bash
cd ACHRcu
./ACHRCuda ecoli_core.mps ecoli_core200warmup.csv 2 1000 1000
```

## Output format

Each output file (`File0`, `File1`, ...) is a space-separated matrix with:
- **Rows**: reactions (in the same order as the MPS file)
- **Columns**: sampled flux points

You can load the results in MATLAB:

```matlab
data = dlmread('File0');
```

Or in Python:

```python
import numpy as np
data = np.loadtxt('File0')
```

