#!/usr/bin/env python
"""
Summary
------------

Gradient Descent for Dynamic mapping

"""

import numpy as np
import healpy as hp
import pylab 
import matplotlib.pyplot as plt
import time
import matplotlib
import sys
import tqdm
import scipy

from sot.core import mocklc 
from sot.core import sepmat 
from sot.core import mvmap
from sot.dymap import gpkernel 
from sot.optim import fdysot
from sot.sotplot import plotdymap

if __name__ == "__main__":

    fontsize=16
    matplotlib.rcParams.update({'font.size':fontsize})
    
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

    Nsample=10000
    tau=360.0
    gamma=16.5/180.0*np.pi
    sep=sepmat.calc_sepmatrix(nside)
    KS=gpkernel.RBF(sep,gamma)
    KT=gpkernel.Matern32(obst,tau)
    invKS=np.linalg.inv(KS)
    invKT=np.linalg.inv(KT)
    alpha=0.25

    eta=0.001    
    M=np.random.normal(loc=0.0,scale=1.0,size=np.shape(W))
    for i in tqdm.tqdm(range(0,Nsample)):
#    for i in range(0,Nsample):
        #residual vector
        rL=fdysot.residual(lc,W,M)
        #dRMdM=fdysot.dRMdM(M,invKS,invKT,alpha)
        dRMdM = 2.0*M
        dLdM=fdysot.dLdM(W,rL)/sigma**2
        dQdM=dLdM+dRMdM
        M = M - eta*dQdM

    frames=[0,int(Ni/2),Ni-1] 
    plotdymap.plotseqmap(M,frames,"mapest","",vmin=0.0,vmax=1.3)
