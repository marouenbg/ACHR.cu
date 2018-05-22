This repo contains the code for ACHR.cu the cuda GPU implementation of the sampling algorithm ACHR.

The repo also contains VFwarmup which an implementation of the generation of sampling warmup points using dynamic load balancing.

Sampling of the solution space of metabolic models is a two-step process:

1- Generation of warmup points

2- The actual sampling of the solution space starting from the warmup points.


| Model         | CreateWarmupMATLAB | CreateWarmupVF  |VF  |VF |VF |VF |VF |
| ------------- |:------------------:| ---------------:|----|---|---|---|---|
| Cores         | 1                  | 1               |2   |4  |8  |16 |32 |
| Ecoli core    |149                 |2.8              |1.8 |0.8|0.7|0.5|0.5|
| P Putida      | 385                | 12.5            |13  |8  |4  |2  |2  |
| EcoliK12      | 801                |    49           |43  |23 |10.4|9.5|9.1|
| Recon2        | 11346              |     288         |186 |30 |32  |24 |21|
| E Matrix      | NA                 |   602           |508 |130|52  |43 |43|
| Ec Matrix     | NA                 | 5275            |4986|924|224 |118|117|
