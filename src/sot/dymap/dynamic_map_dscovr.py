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
import rundynamic_cpu as rundynamic
import plotdymap
import process_dscovr as prds

#from multiprocessing import Pool

#from numpy.random import standard_normal

if __name__ == "__main__":

    fontsize=16
    matplotlib.rcParams.update({'font.size':fontsize})
    
    # LOAD sampled parameters
    fsample="samples/test531/flat_sample_dscovrRBF.npz"
    dat=np.load(fsample,allow_pickle=True)
    flat_samples=dat["arr_0"]
    print(np.shape(flat_samples))
    flat_samples_corner=np.copy(flat_samples)
    flat_samples_corner[:,0]=flat_samples[:,0]/np.pi*180 #rad2deg

    W, obst, lc, timeobj=prds.lcdscovr("pca",Nreduce=2,Nfin=2435)

    W=dat["arr_1"]
    lc=dat["arr_2"]
    ########################################
    #POSTERIOR
    ########################################
    
    labels=["$\\gamma$ (deg)","$\\alpha$","$\\tau$ (d)","sigma"]
    tag="RBF_531"
    import corner
    fig = corner.corner(flat_samples_corner, labels=labels, truths=[None,None,None,None],truth_color="C1")
    plt.savefig("corner_dscovr"+tag+".png")
    plt.savefig("corner_dscovr"+tag+".pdf")
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
    
#    frames=[78,768,1722]
    frames=np.array(range(0,Ni))
    randmap=[]
    Aast=np.zeros((Ni,Nj))
    for n in tqdm.tqdm(range(0,Nsample)):
        gamma,alpha,tau,sigma=flat_samples[n,:]
        KS=gpkernel.RBF(sep,gamma)
        KT=gpkernel.Matern32(obst,tau)
        Pid=sigma**-2*np.eye(Ni) #Data precision matrix
        Sigmad=sigma**2*np.eye(Ni) 
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
    np.savez("dymap"+tag,Aast_mean)    
    randmap=np.array(randmap)
    np.savez("dyran"+tag,randmap)
    plotdymap.plotseqmap(Aast_mean,frames,"mapest","",vmin=0.0,vmax=1.3)




#plotdymap.plotseqmap(np.diagonal(randmap),frames,"mapran","",vmin=0.0,vmax=1.3)
