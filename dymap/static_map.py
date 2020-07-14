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
import tqdm
import runstatic_cpu as runstatic
#fontsize=20
#matplotlib.rcParams.update({'font.size':fontsize})

# LOAD sampled parameters
fsample="flat_sampleRBF.npz"
dat=np.load(fsample,allow_pickle=True)
flat_samples=dat["arr_0"]
W=dat["arr_1"]
lc=dat["arr_2"]
inputgeo=dat["arr_3"]
inc,Thetaeq,zeta,Pspin,Porb,obst=inputgeo
labels=["zeta","Thetaeq","gamma","alpha"]
tag="RBF"

import corner
Thetaeq=np.pi
zeta=23.4/180.0*np.pi
fig = corner.corner(flat_samples, labels=labels, truths=[zeta,Thetaeq,None,None])
plt.savefig("corner"+tag+".png")
plt.savefig("corner"+tag+".pdf")

sigma=np.mean(lc)*0.01


## kernel
Ni,Nj=np.shape(W)
nside=hp.npix2nside(Nj)
sep=sepmat.calc_sepmatrix(nside)

## comute
Ns,Npar=np.shape(flat_samples)
mu=[]
Nave=100
#Nave=Ns
for i in tqdm.tqdm(range(0,Nave)):
    zeta,Thetaeq,gamma,alpha=flat_samples[i,:]
    KS=alpha*gpkernel.RBF(sep,gamma)
    Pid=sigma**-2*np.eye(Ni) #Data precision matrix
    mueach=runstatic.Mean_STSOT(W,KS,lc,Pid)
    mu.append(mueach)
    
mu=np.array(mu)
mu=np.mean(mu,axis=0)
hp.mollview(mu, title="",flip="geo",cmap=plt.cm.pink,min=0,max=1.0)
plt.savefig("mu.pdf")
plt.show()
