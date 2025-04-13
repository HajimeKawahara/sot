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
from multiprocessing import Pool
from multiprocessing import cpu_count
import read_dscovr as rd
import process_dscovr as prds

#from numpy.random import standard_normal

if __name__ == "__main__":
    fontsize=16
    matplotlib.rcParams.update({'font.size':fontsize})

    outdir="dscovr"
    fsample="samples/test529/flat_sample_dscovrRBF.npz"
    dat=np.load(fsample,allow_pickle=True)
    flat_samples=dat["arr_0"]
    W=dat["arr_1"]

    Ns,Npar=np.shape(flat_samples)
    Nsample=Ns
    Ni,Nj=np.shape(W)


    ###
    Nf=6
    
    ##### should be corrected
    W, obst, lc, timeobj=prds.lcdscovr("pca",Nreduce=2,Nfin=2435)
    frames=np.array(range(0,Ni))
    frames=frames[::Nf]
    timelab=True
    if timelab:
        timeframe=timeobj[frames]
    title=[]
    for tt in timeframe:
        title.append(str(tt.strftime("%Y-%m-%d")))
    title=np.array(title)
    ##########
    ## compute

    Aast=[]
    randmap=[]   
    frames=np.array(range(0,Ni))
    frames=frames[::Nf]
    dat=np.load("dymap_dscovrmRBF.npz")
    Aast=dat["arr_0"]
#    Aast_mean=(Aast)/Nsample
    plotdymap.plotseqmap(Aast,frames,outdir+"/map",Earth=True,vmin=-0.9,vmax=0.2,title=title)



#plotdymap.plotseqmap(np.diagonal(randmap),frames,"mapran","",vmin=0.0,vmax=1.3)
