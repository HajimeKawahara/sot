<img src="https://github.com/HajimeKawahara/sot/blob/master/data/fig/logo.png" Titie="Spin-Orbit Tomography" Width=600px>

In the near future, direct imaging missions will search for Earth-like planets around nearby stars. One of the problems is how to characterize the planet surface. To address this question, we are developing a surface map and components reconstruction method using a one-dimensional light curve of a direct-imaged planet. The orbital motion and spin rotation of a planet conveys information about the spherical surface to the time-series of the light curve. In the future, this theoretical work will be tested in the era of space direct imaging of exoplanets.

## Spin-Orbit Unmixing 
Spin-Orbit Unmixing (SOU) is a unified retrieval model for spectral unmixing and spin-orbit tomography.

### NMF with Volume Regularization 
Spin-Orbit Unmixing using the [non-negative matrix factorization (NMF)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) and L2 and volume regularization (SOU-NMF). 

- sot/nmfmap/nmfsot.py
This code solves SOU-NMF for a cloudless toy model.

The algorithm is based on [Kawahara (2020)]().


<img src="https://github.com/HajimeKawahara/sot/blob/master/data/fig/sotnmf.png" Titie="The recovered composite map of the real light curve of Earth by DSCOVR using SOU-NMF" Width=300px>

The recovered composite map of the real light curve of Earth by 
 [Deep Space Climate Observatory](https://en.wikipedia.org/wiki/Deep_Space_Climate_Observatory) 
using SOU-NMF.

## Spin-Orbit Tomography
Spin-Orbit Tomography (SOT) is a retrieval technique of a 2 dimensinal map of an Exo Earth from time-series data of integrated reflection light.

### SOT + Sparse Modeling
SOT-Sparse uses L1 and Total Squared Variation (TSV).

- [Spare](https://github.com/2ndmk2/Spare) 

The algorithm is based on [Aizawa, Kawahara, Fan (2020)]().

### PyTorch Version
SOT-L2 using automatic differentiation and ADAM optimizer in PyTorch. 

- sot/sotorch

### L2/Tikhonov Regularization
This jupyter notebook includes SOT + L2 regularization (SOT-L2) and L-curve criterion.

- sot/l2map.ipynb

The algorithm is based on [Kawahara & Fujii (2010)](https://arxiv.org/abs/1004.5152),[Kawahara & Fujii (2011)](http://arxiv.org/abs/1106.0136), and [Fujii & Kawahara (2012)](http://arxiv.org/abs/1204.3504).

## Frequency Modulation
The orientation of the spin axis can be inferred from frequency modulation (FM) of the light curve. 

The algorithm is based on [Kawahara (2016)](https://arxiv.org/abs/1603.02898).

## Related Projects

- [ReflectDirect](https://github.com/joelcolinschwartz/ReflectDirect) Python suite for analysis of planet reflected light by Joel Schwartz et al.
- [exocartographer](https://github.com/bfarr/exocartographer) Bayesian framework of a 2D mapping and obliquity measurement by Ben Farr et al.
- [EARL](https://github.com/HalHaggard/EARL) Spherical harmonics decomposition of reflected light by Hal Hagaard et al.
- [starry](https://github.com/rodluger/starry) Tools for mapping planets and stars by Rodrigo Luger et al.
- [bluedot](https://github.com/HajimeKawahara/bluedot) A pale blue dot simulater by Hajime Kawahara. 
