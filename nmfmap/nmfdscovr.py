import time
import numpy as np
import io_surface_type 
import io_refdata
import toymap
import mocklc
import matplotlib.pyplot as plt
import healpy as hp
import cupy as cp
import sys
import healpy as hp
import initnmf
import runnmf
import read_dscovr as rd

np.random.seed(34)

## load DSCOVR
nside=16
npix=hp.nside2npix(nside)
W,t,lcall=rd.read_dscovr("/home/kawahara/exomap/data/for_HKawahara",9)

noiselevel=0.0001
lcall=lcall+noiselevel*np.mean(lcall)*np.random.normal(0.0,1.0)

print(np.min(lcall))
print(len(lcall[lcall<0]))
##################################################

#normmat=np.diag(1.0/np.sum(lcall,axis=0))
N=3
lamA=3.e-4
lamX=3.e-2

## NMF Initialization ============================
#A0,X0=initnmf.init_random(N,npix,lcall)
A0,X0=initnmf.initpca(N,W,lcall,lamA)
runnmf.LogNMF(-1,A0,X0,3)

Ntry=10000000
#lamA=0.0
#lamX=0.0
epsilon=1.e-16

#A,X=runnmf.NG_MVC_NMF(Ntry,lcall,W,A0,X0,lam,epsilon)
#A,X=runnmf.NG_L2MVC_NMF(Ntry,lcall,W,A0,X0,lam,epsilon)
A,X,logmetric=runnmf.L2_NMF(Ntry,lcall,W,A0,X0,lamA,lamX,epsilon)
#A,X=runnmf.QP_MVC_NMF(Ntry,lcall,W,A0,X0,lam,epsilon)

np.savez("ax_dscovr"+str(int(np.log10(lamX)))+"_"+str(int(np.log10(lamA))),A,X)
np.savez("metric_dscovr"+str(int(np.log10(lamX)))+"_"+str(int(np.log10(lamA))),logmetric)
#plt.show()
