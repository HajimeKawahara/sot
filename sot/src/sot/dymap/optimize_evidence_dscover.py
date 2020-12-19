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
import unmixed_curve as uc
import process_dscovr as prds

mode="pca"
W, obst, lc, timeobj=prds.lcdscovr("pca",Nreduce=2,Nfin=2435)

Ni=len(lc)
Ns=300

gam0=0.1
alpha0=0.5
tau0=50.0
sigma0=0.01

gam0_acute=np.log(gam0)
tau0_acute=np.log(tau0)
alpha0_acute=np.log(alpha0)
sigma0_acute=np.log(sigma0)

x0=[gam0_acute,alpha0_acute,tau0_acute,sigma0_acute]

nside=16
npix=hp.nside2npix(nside)

mode=True
#unmix

### FRAME for png/pdf
frames=np.array(range(0,Ni))
frames=frames[::10]
timeframe=timeobj[frames]

title=[]
for tt in timeframe:
    title.append(str(tt.strftime("%Y-%m-%d")))
title=np.array(title)
## RBF kernel
nside=16
npix=hp.nside2npix(nside)
sep=sepmat.calc_sepmatrix(nside)

## evidence optimization

tag="RBF"


def func(x):
    gam_acute,alpha_acute,tau_acute,sigma_acute=x
    tau=np.exp(tau_acute)
    alpha=np.exp(alpha_acute)
    gamma=np.exp(gam_acute)
    sigma=np.exp(sigma_acute)
    print(tau,alpha,gamma,sigma)
    KS=gpkernel.RBF(sep,gamma)
    KT=gpkernel.Matern32(obst,tau=tau)
    WSWT=(W@KS@W.T)
    Kw=alpha*KT*(WSWT)
    covd=sigma**2*np.eye(Ni)

    Cov = covd + Kw
    sign,logdet=np.linalg.slogdet(Cov)
    try:
        Xlc=scipy.linalg.solve(Cov,lc,assume_a="pos")
        nlev = logdet+lc@Xlc
    except:
        nlev=np.inf

    return nlev

def jac(x):
    gam_acute,alpha_acute,tau_acute,sigma_acute=x
    tau=np.exp(tau_acute)
    alpha=np.exp(alpha_acute)
    gamma=np.exp(gam_acute)
    sigma=np.exp(sigma_acute)

    KS=gpkernel.RBF(sep,gamma)
    KT=gpkernel.Matern32(obst,tau=tau)
    WSWT=(W@KS@W.T)
    Kw=alpha*KT*(WSWT)
    covd=sigma**2*np.eye(Ni)
    Cov = covd + Kw
    sign,logdet=np.linalg.slogdet(Cov)
    P=scipy.linalg.inv(Cov)
    
    Xlc=P@lc
    
    ## gamma
    dgam_acute=gpkernel.d_RBF(sep,gamma)
    dKw=alpha*KT*(W@dgam_acute@W.T)
    dLdgam=np.trace(P@dKw)-Xlc@dKw@Xlc
    ## alpha
    dLdalpha=np.trace(P@Kw)-Xlc@Kw@Xlc
    ## tau
    dtau_acute=gpkernel.d_Matern32(obst,tau)
    dKw=alpha*dtau_acute*(WSWT)
    dLdtau=np.trace(P@dKw)-Xlc@dKw@Xlc
    ##
    dLdsigma=np.trace(P@covd)-Xlc@covd@Xlc
    
    grad=np.array([dLdgam,dLdalpha,dLdtau,dLdsigma])
    return grad

ts=time.time()
#res=scipy.optimize.minimize(func, x0,method="powell")
res=scipy.optimize.minimize(func,x0,method="L-BFGS-B",jac=jac,tol=1.e-12)
gamast=np.exp(res["x"][0])
alphaast=np.exp(res["x"][1])
tauast=np.exp(res["x"][2])
sigmaast=np.exp(res["x"][3])

print("gamma,alpha,tau,sigma=",gamast/np.pi*180,"deg",alphaast,tauast,"d",sigmaast)
print("gamma,alpha,tau,sigma=",gamast,"radian",alphaast,tauast,"d",sigmaast)

# Final map
import rundynamic_cpu as rundynamic
import plotdymap
KS=gpkernel.RBF(sep,gamast)
KT=gpkernel.Matern32(obst,tauast)
Pid=sigmaast**-2*np.eye(Ni) #Data precision matrix
Aast=rundynamic.Mean_DYSOT(W,KS,KT,alphaast,lc,Pid)

plotdymap.plotseqmap(Aast,frames,"mapev",title=title,Earth=True)#,vmin=0.05,vmax=0.35)

