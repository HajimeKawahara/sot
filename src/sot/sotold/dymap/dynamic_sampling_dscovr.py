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
import process_dscovr as prds

Ns=2000

mode="pca"
W, obst, lc, timeobj=prds.lcdscovr("pca",Nreduce=2,Nfin=2435)
Ni,Nj=np.shape(W)

gam0=5.38179711e-01
alpha0=2.41633818e-01
tau0=2.41896443e+01
sigma0=3.73163901e-03
#gam0=0.45320403586138686
#alpha0=0.9467923937650575
#tau0=43.324518581987334
#sigma0=0.004436772207492778

## RBF kernel
nside=16
npix=hp.nside2npix(nside)
sep=sepmat.calc_sepmatrix(nside)

## optimization

tag="RBF"
## spin and hyperparameter MCMC sampling using emcee
def log_prior(theta):
    p_gamma,p_alpha,p_tau,p_sigma=theta
    if 1.e-10 < p_gamma < np.pi/3.0 and 1.e-10 < p_alpha and 1.e-4 < p_tau < 1.e4 and 1.e-5 < p_sigma < 1.e5:
        return np.log(1.0/p_alpha/p_gamma/p_tau/p_sigma)
    return -np.inf

def log_likelihood(theta, d):
    p_gamma,p_alpha,p_tau,p_sigma=theta
    #KS=p_alpha*gpkernel.Matern32(sep,p_gamma)
    KS=gpkernel.RBF(sep,p_gamma)
    KT=gpkernel.Matern32(obst,tau=p_tau)
    WSWT=(W@KS@W.T)
    Kw=p_alpha*KT*(WSWT)
    ######
    covd=p_sigma**2*np.eye(Ni)
    Cov = covd + Kw
    try:
        sign,logdet=np.linalg.slogdet(Cov)
        Pi_d=scipy.linalg.solve(Cov,d,assume_a="pos")
        prop = -0.5*logdet-0.5*d@Pi_d #-0.5*np.shape(cov)[0]*np.log(2.0*np.pi)
    except:
        prop = -np.inf
        
    return prop

def log_probability(theta, d):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, d)



pos = np.array([gam0,alpha0,tau0,sigma0])+ 1e-4 * np.random.randn(32, 4)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(lc,))
sampler.run_mcmc(pos, Ns, progress=True);

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
#samples = sampler.get_chain()
#print(samples)
labels=["gamma","alpha","tau","sigma"]

np.savez("flat_sample_dscovr"+tag,flat_samples,W,lc)

import corner
fig = corner.corner(flat_samples, labels=labels, truths=[zeta,Thetaeq,None,None,None])
plt.savefig("corner_dscovr"+tag+".png")
plt.savefig("corner_dscovr"+tag+".pdf")
plt.show()
