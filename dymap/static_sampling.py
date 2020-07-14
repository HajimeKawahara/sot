#!/usr/bin/env python

import numpy as np
import healpy as hp
import pylab 
import matplotlib.pyplot as plt
import time
import mocklc 
import matplotlib
import sepmat 
import gpkernel 
import scipy
import emcee
import sys
import time

Ns=2000

np.random.seed(17)

#set geometry
inc=45.0/180.0*np.pi
Thetaeq=np.pi
zeta=60.0/180.0*np.pi
Pspin=23.9344699/24.0 #Pspin: a sidereal day
wspin=2*np.pi/Pspin
Porb=365.242190402                                            
worb=2*np.pi/Porb                                                                                                                
Ni=1024
obst=np.linspace(0.0,Porb,Ni)

# test map
nside=16
npix=hp.nside2npix(nside)
mmap=hp.read_map("/home/kawahara/exomap/sot/data/mockalbedo16.fits")
mask=(mmap>0.0)
mmap[mask]=1.0
M=len(mmap)

#generating light curve
Thetav=worb*obst
Phiv=np.mod(wspin*obst,2*np.pi)
WI,WV=mocklc.comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]
lc=np.dot(W,mmap)

sigma=np.mean(lc)*0.1
noise=np.random.normal(0.0,sigma,len(lc))
lc=lc+noise

## RBF kernel
nside=16
npix=hp.nside2npix(nside)
sep=sepmat.calc_sepmatrix(nside)

## optimization

tag="RBFobl"
## spin and hyperparameter MCMC sampling using emcee
def log_prior(theta):
    p_zeta,p_Thetaeq,p_gamma,p_alpha=theta
    if 0.0 < p_zeta < np.pi and 0.0 < p_Thetaeq < 2*np.pi and 1.e-10 < p_gamma < np.pi/3.0 and 1.e-10 < p_alpha:
        return np.log(np.sin(p_zeta)/p_alpha/p_gamma)
    return -np.inf

def log_likelihood(theta, d, covd):
    p_zeta,p_Thetaeq,p_gamma,p_alpha=theta
    WI,WV=mocklc.comp_weight(nside,p_zeta,inc,p_Thetaeq,Thetav,Phiv)
    Wp=WV[:,:]*WI[:,:]
    #KS=p_alpha*gpkernel.Matern32(sep,p_gamma)
    KS=p_alpha*gpkernel.RBF(sep,p_gamma)
    Cov = covd + Wp@KS@Wp.T
    sign,logdet=np.linalg.slogdet(Cov)
    Pi_d=scipy.linalg.solve(Cov,d,assume_a="pos")
    prop = -0.5*logdet-0.5*d@Pi_d #-0.5*np.shape(cov)[0]*np.log(2.0*np.pi)
    return prop

def log_probability(theta, d, covd):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, d, covd)

gam0=0.29298260376811
alpha0=sigma**2*0.774263682681127
pos = np.array([zeta,Thetaeq,gam0,alpha0])+ 1e-4 * np.random.randn(16, 4)
nwalkers, ndim = pos.shape

#Assumming we know the data covariance
covd=sigma**2*np.eye(Ni)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(lc, covd))
sampler.run_mcmc(pos, Ns, progress=True);

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
#samples = sampler.get_chain()
#print(samples)
labels=["zeta","Thetaeq","gamma","alpha"]

inputgeo=[inc,Thetaeq,zeta,Pspin,Porb,obst]
np.savez("flat_sample"+tag,flat_samples,W,lc,inputgeo)

import corner
fig = corner.corner(flat_samples, labels=labels, truths=[zeta,Thetaeq,None,None])
plt.savefig("corner_"+tag+".png")
plt.savefig("corner_"+tag+".pdf")
plt.show()
