#!/usr/bin/env python
"""
Summary
------------

Dynamic mapping using a geometric sampling (dynamic_sampling.py)

"""

import numpy as np
import healpy as hp
import pylab 
import matplotlib.pyplot as plt
import time
import matplotlib
import emcee
import sys
import tqdm
import scipy

from sot.core import mocklc 
from sot.core import sepmat 
from sot.dymap import gpkernel 
from sot.dymap import rundynamic_cpu as rundynamic
from sot.sotplot import plotdymap
#from multiprocessing import Pool

#from numpy.random import standard_normal

if __name__ == "__main__":

    fontsize=16
    matplotlib.rcParams.update({'font.size':fontsize})
    
    # LOAD sampled parameters
    fsample="/home/kawahara/exomap/gpmap/samples/test524/flat_sample_dyRBF.npz"
    dat=np.load(fsample,allow_pickle=True)
    flat_samples=dat["arr_0"]
    print(np.shape(flat_samples))
    flat_samples_corner=np.copy(flat_samples)
    flat_samples_corner[:,0:3]=flat_samples[:,0:3]/np.pi*180 #rad2deg

    
    W=dat["arr_1"]
    lc=dat["arr_2"]
    inputgeo=dat["arr_3"]
    inc,Thetaeq,zeta,Pspin,Porb,obst=inputgeo

    worb=2*np.pi/Porb                                                          
    Thetav=worb*obst
    wspin=2*np.pi/Pspin
    Phiv=np.mod(wspin*obst,2*np.pi)

    sigma=np.mean(lc)*0.01
    ########################################
    #POSTERIOR
    ########################################
    
    labels=["$\\zeta$ (deg)","$\\Theta_\\mathrm{eq}$ (deg)","$\\gamma$ (deg)","$\\alpha$","$\\tau$ (d)"]
    tag="RBFnew"
    import corner
    fig = corner.corner(flat_samples_corner, labels=labels, truths=[zeta/np.pi*180,Thetaeq/np.pi*180,None,None,None],truth_color="C1")
    plt.savefig("corner_dy"+tag+".png")
    plt.savefig("corner_dy"+tag+".pdf")
    #plt.show()
    
    ########################################
    #MEAN MAP
    ########################################
    
    Ni,Nj=np.shape(W)
    nside=hp.npix2nside(Nj)
    sep=sepmat.calc_sepmatrix(nside)
    
    ## compute
    Ns,Npar=np.shape(flat_samples)
    Aast=[]
    #Nsample=10
    Nsample=Nj
    randmap=[]
    rm=False
    
    frames=[0,int(Ni/2),Ni-1] 
    randmap=[]
    Aast=np.zeros((Ni,Nj))
    for n in tqdm.tqdm(range(0,Nsample)):
        zeta,Thetaeq,gamma,alpha,tau=flat_samples[n,:]
        KS=gpkernel.RBF(sep,gamma)
        KT=gpkernel.Matern32(obst,tau)
        Pid=sigma**-2*np.eye(Ni) #Data precision matrix
        Sigmad=sigma**2*np.eye(Ni) 

        #weight recomputing
        WI,WV=mocklc.comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
        W=WV[:,:]*WI[:,:]

        #Aasteach=rundynamic.Mean_DYSOT(W,KS,KT,alpha,lc,Pid)
        P,Aasteach=rundynamic.PMean_DYSOT(W,KS,KT,alpha,lc,Pid,Sigmad)
        Aast=Aast+Aasteach
        
        #RANDOMIZED MAP
        randmap_each=[]
        for i in frames:
            snapcov=rundynamic.Covi_snap(W,KS,KT,alpha,P,i)
            snapshot=Aasteach[i,:]
            randmap_each.append(np.random.multivariate_normal(snapshot,snapcov,1,check_valid="ignore"))
        randmap.append(randmap_each)

    Aast_mean=(Aast)/Nsample
    plotdymap.plotseqmap(Aast_mean,frames,"mapest","",vmin=0.0,vmax=1.3)
    np.savez("dymap"+tag,Aast_mean)
    randmap=np.array(randmap)
    np.savez("dyran"+tag,randmap)


