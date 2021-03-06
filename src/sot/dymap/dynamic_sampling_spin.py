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
import mvmap

Ns=2000

np.random.seed(53)

#set geometry
inc=45.0/180.0*np.pi
Thetaeq=np.pi
zeta=23.4/180.0*np.pi
Pspin=23.9344699/24.0 #Pspin: a sidereal day
wspin=2*np.pi/Pspin
Porb=365.242190402                                            
worb=2*np.pi/Porb                                                                                                                
Ni=1024
obst=np.linspace(0.0,Porb,Ni)

# test moving map
nside=16
npix=hp.nside2npix(nside)
mmap=hp.read_map("/home/kawahara/exomap/sot/data/mockalbedo16.fits")
mask=(mmap>0.0)
mmap[mask]=1.0
M=mvmap.rotating_map(mmap,obst,rotthetamax=np.pi/2.0)
#geometric weight
ts=time.time()
Thetav=worb*obst
Phiv=np.mod(wspin*obst,2*np.pi)
WI,WV=mocklc.comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]
print("Weight",time.time()-ts,"sec")

#Light curve
ts=time.time()
lc=np.sum(W*M,axis=1)
noiselevel=0.01
sigma=noiselevel*np.mean(lc)
noise=sigma*np.random.normal(0.0,1.0,np.shape(lc))
lc=lc+noise
print("Lc",time.time()-ts,"sec")

## RBF kernel
nside=16
npix=hp.nside2npix(nside)
sep=sepmat.calc_sepmatrix(nside)

## optimization

tag="RBFspin"
## spin and hyperparameter MCMC sampling using emcee
def log_prior(theta):
    p_zeta,p_Thetaeq,p_gamma,p_alpha,p_tau,p_Pspin=theta
    if 0.0 <= p_zeta <= np.pi and 0.0 <= p_Thetaeq <= 2*np.pi and 0.01 <= p_gamma <= np.pi/2.0 and 1.e-4 <= p_alpha <= 1.e4 and 1.e-4 <= p_tau <= 1.e4 and 0.5 < p_Pspin < 1.5 :
        return np.log(np.sin(p_zeta)/p_alpha/p_gamma/p_tau/p_Pspin)
    return -np.inf

def log_likelihood(theta, d, covd):
    p_zeta,p_Thetaeq,p_gamma,p_alpha,p_tau,p_Pspin=theta
    wspin=2*np.pi/p_Pspin
    Phiv=np.mod(wspin*obst,2*np.pi)
    WI,WV=mocklc.comp_weight(nside,p_zeta,inc,p_Thetaeq,Thetav,Phiv)
    Wp=WV[:,:]*WI[:,:]
    #KS=p_alpha*gpkernel.Matern32(sep,p_gamma)
    KS=gpkernel.RBF(sep,p_gamma)
    KT=gpkernel.Matern32(obst,tau=p_tau)
    WSWT=(Wp@KS@Wp.T)
    Kw=p_alpha*KT*(WSWT)
    ######
    Cov = covd + Kw
    try:
        sign,logdet=np.linalg.slogdet(Cov)
        Pi_d=scipy.linalg.solve(Cov,d,assume_a="pos")
        prop = -0.5*logdet-0.5*d@Pi_d #-0.5*np.shape(cov)[0]*np.log(2.0*np.pi)
        return prop
    except:
        return -np.inf

def log_probability(theta, d, covd):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, d, covd)

gam0=0.24530731755686958
alpha0=sigma**2*0.5501245233258051
tau0=375.6520066482577
Pspin0=Pspin

pos = np.array([zeta,Thetaeq,gam0,alpha0,tau0,Pspin0])+ 1e-4 * np.random.randn(32, 6)
nwalkers, ndim = pos.shape

#Assumming we know the data covariance
covd=sigma**2*np.eye(Ni)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(lc, covd))
sampler.run_mcmc(pos, Ns, progress=True);

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
#samples = sampler.get_chain()
#print(samples)
labels=["zeta","Thetaeq","gamma","alpha","tau","pspin"]

inputgeo=[inc,Thetaeq,zeta,Pspin,Porb,obst]
np.savez("flat_sample_dy"+tag,flat_samples,W,lc,inputgeo)

import corner
fig = corner.corner(flat_samples, labels=labels, truths=[zeta,Thetaeq,None,None,None,Pspin])
plt.savefig("corner_dy"+tag+".png")
plt.savefig("corner_dy"+tag+".pdf")
plt.show()
