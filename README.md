<img src="https://github.com/HajimeKawahara/sot/blob/master/data/fig/logo.png" Titie="Spin-Orbit Tomography" Width=600px>

In the near future, direct imaging missions will search for Earth-like planets around nearby stars. One of the problems is how to characterize the planet surface. To address this question, we are developing a surface map and components reconstruction method using a one-dimensional light curve of a direct-imaged planet. The orbital motion and spin rotation of a planet conveys information about the spherical surface to the time-series of the light curve. In the future, this theoretical work will be tested in the era of space direct imaging of exoplanets.

## Spin-Orbit Unmixing 
Spin-Orbit Unmixing (SOU) is a unified retrieval model for spectral unmixing and spin-orbit tomography.

### NMF with Volume Regularization 
Spin-Orbit Unmixing using the [non-negative matrix factorization (NMF)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) and L2 and volume regularization (SOU-NMF). 

- sot/nmfmap/nmfsot.py
This code solves SOU-NMF for a cloudless toy model.

The algorithm is based on [Kawahara (2020)](http://arxiv.org/abs/2004.03931) (ApJ accepted).

<img src="https://github.com/HajimeKawahara/sot/blob/master/data/fig/sotnmf.png" Titie="The recovered composite map of the real light curve of Earth by DSCOVR using SOU-NMF" Width=300px>

The recovered composite map of the real light curve of Earth by 
 [Deep Space Climate Observatory](https://en.wikipedia.org/wiki/Deep_Space_Climate_Observatory) 
using SOU-NMF.

## Spin-Orbit Tomography
Spin-Orbit Tomography (SOT) is a retrieval technique of a 2 dimensinal map of an Exo Earth from time-series data of integrated reflection light.

### Dynamic Mapping

SOT for time-varying geometry. Coming soon.

### SOT + Sparse Modeling
SOT-Sparse uses L1 and Total Squared Variation (TSV).

- [Spare](https://github.com/2ndmk2/Spare) 

The algorithm is based on [Aizawa, Kawahara, Fan (2020)](https://arxiv.org/abs/2004.03941).

### L2/Tikhonov Regularization

Jupyter notebooks for the SOT + L2 regularization (SOT-L2), L-curve criterion, including the code from scratch, with use of scikit-learn.Ridge, and with the optimization using [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) and ADAM optimizer in PyTorch.

- sot/tutorial

The algorithm is based on [Kawahara & Fujii (2010)](https://arxiv.org/abs/1004.5152),[Kawahara & Fujii (2011)](http://arxiv.org/abs/1106.0136), and [Fujii & Kawahara (2012)](http://arxiv.org/abs/1204.3504).

## Frequency Modulation
The orientation of the spin axis can be inferred from frequency modulation (FM) of the light curve. 

- fm/rottheory.py the modulation factor. It can reproduce Figure 2 in [Kawahara (2016)](https://arxiv.org/abs/1603.02898).

<img src="https://github.com/HajimeKawahara/sot/blob/master/data/fig/rott.png" Titie="Fig 2 in Kawahara 2016" Width=270px><img src="https://github.com/HajimeKawahara/sot/blob/master/data/fig/rott2.png" Titie="Fig 2 in Kawahara 2016" Width=270px><img src="https://github.com/HajimeKawahara/sot/blob/master/data/fig/rott3.png" Titie="Fig 2 in Kawahara 2016" Width=270px>

- [juwvid](https://github.com/HajimeKawahara/juwvid) Code for the Wigner-Ville analysis written in Julia-0.6.

The algorithm is based on [Kawahara (2016)](https://arxiv.org/abs/1603.02898).

## USER-UNFRIENDLY INSTALL...

Set PYTHONPATH = /location/sot/sot/core

Set PYTHONPATH = /location/sot/sot/plot

Set PYTHONPATH = /location/sot/nmfmap

Install some python modules you got in error messages.

## Related Projects

- [ReflectDirect](https://github.com/joelcolinschwartz/ReflectDirect) Python suite for analysis of planet reflected light by [Joel Schwartz et al](https://arxiv.org/abs/1511.05152).
- [exocartographer](https://github.com/bfarr/exocartographer) Bayesian framework of a 2D mapping and obliquity measurement by [Ben Farr et al](https://arxiv.org/abs/1802.06805).
- [EARL](https://github.com/HalHaggard/EARL) Spherical harmonics decomposition of reflected light by [Hal Haggard et al](https://arxiv.org/abs/1802.02075).
- [samurai](https://github.com/jlustigy/samurai) Rotational Unmixing by [Lustig-Yaeger et al](https://arxiv.org/abs/1901.05011).
- [starry](https://github.com/rodluger/starry) Tools for mapping planets and stars by [Rodrigo Luger et al](https://arxiv.org/abs/1903.12182).
- [bluedot](https://github.com/HajimeKawahara/bluedot) A pale blue dot simulater by Hajime Kawahara (beta version). 
