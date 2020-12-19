# Bayesian Dynamic Spin-Orbit Tomography

|  function   |      |
| ----------- | ---- |
|  Geography  |  o   |
|  Unmixing   |  -   |
|  Dynamic    |  o   |
|  Bayesian   |  o   |


## Bayesian Dynamic SOT for a Toy Model 

- dynamic_sampling.py: sampling of the posterior of the nonlinear parameters 
- dynamic_map.py: sampling and mean of the geography

[![Dynamic map (toy model)](https://img.youtube.com/vi/eP-aQ0PVPAs/0.jpg)](https://youtu.be/eP-aQ0PVPAs)

Figure (Click): Input toy model and the retrieved dynamic map.

## Same but with Spin Rate as a Free Parameter

- dynamic_sampling_spin.py: sampling of the posterior of the nonlinear parameters 
- dynamic_map_spin.py: sampling and mean of the geography

## DSCOVR

- dynamic_sampling_dscovr.py: sampling of the posterior of the nonlinear parameters 
- dynamic_map_dscovr.py: sampling and mean of the geography

## Bayesian Static SOT

- static_sampling.py: sampling of the posterior of the nonlinear parameters 
- static_map.py: sampling and mean of the geography

## Analysis of the MODIS Cloud Fraction Data

- analyzeMYD.py

## Drivers

- gpkernel.py
- gpmatrix.py
- rundynamic_cpu.py
- runstatic_cpu.py

## Others

- process_dscovr.py
- optimize_evidence_dscover.py
- plottings
-- plotdymap.py
