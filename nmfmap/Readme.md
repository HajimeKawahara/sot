# Spin-Orbit Unmixing using Nonnegative Matrix Factrization 

|  function   |      |
| ----------- | ---- |
|  Geography  |  o   |
|  Unmixing   |  o   |
|  Dynamic    |  -   |
|  Bayesian   |  -   |


- runnmf_gpu.py Optimizer for the geometric NMF with volume regularization, using cupy.
- runnmf_cpu.py CPU version, slow, less tested than the GPU version.
- initnmf.py Generate various initial conditions.

## main

- nmfsot.py SOU-NMF for simulated lightcerve
- nmfdirect.py Direct NMF for light curve
- nmfdscovr.py SOU-NMF for DSCOVR data (we do not provide the data itself though.)
