<img src="https://github.com/HajimeKawahara/sot/blob/master/data/fig/logo.png" Titie="explanation" Width=300px>

Toward a picture of an exo Earth.

## Spin-Orbit Unmixing 
Spin-Orbit Unmixing (SOU) is a unified retrieval model for spectral unmixing and spin-orbit tomography.

### NMF with Volume Regularization 
Spin-Orbit Unmixing using the [non-negative matrix factorization (NMF)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) and L2 and volume regularization (SOU-NMF). 

- sot/nmfmap/nmfsot.py
This code solves SOU-NMF for a cloudless toy model.

References:
[Kawahara (2020)]()

## Spin-Orbit Tomography
Spin-Orbit Tomography (SOT) is a retrieval technique of a 2 dimensinal map of an Exo Earth from time-series data of integrated reflection light.

### SOT + sparse modeling
SOT-Sparse uses L1 and Total Squared Variation (TSV).

- [Spare](https://github.com/HajimeKawahara/Spare) 

References:
[Aizawa, Kawahara, Fan (2020)]()

### PyTorch version
SOT-L2 using automatic differentiation and ADAM optimizer in Pytorch. 

- sot/sotorch

### L2/Tikhonov regularization
This jupyter notebook includes SOT + L2 regularization (SOT-L2) and L-curve criterion.

- sot/l2map.ipynb

References:
[Kawahara & Fujii (2010)](https://arxiv.org/abs/1004.5152),[Kawahara & Fujii (2011)](http://arxiv.org/abs/1106.0136),[Fujii & Kawahara (2012)](http://arxiv.org/abs/1204.3504)

## Frequency Modulation
The orientation of the spin axis can be inferred from frequency modulation (FM) of the light curve. 

References:[Kawahara (2016)](https://arxiv.org/abs/1603.02898)

## Related projects

- [ReflectDirect](https://github.com/joelcolinschwartz/ReflectDirect)
- [exocartographer](https://github.com/bfarr/exocartographer)
- [EARL](https://github.com/HalHaggard/EARL)
- [starry](https://github.com/rodluger/starry) 
- [bluedot](https://github.com/HajimeKawahara/bluedot) 
