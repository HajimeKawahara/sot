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
import runnmf_gpu as runnmf
import read_dscovr as rd

np.random.seed(34)

## load DSCOVR
nside=16
npix=hp.nside2npix(nside)
W,t,lcall=rd.read_dscovr("/home/kawahara/exomap/data/for_HKawahara",4,istart=3)
lcall=lcall*npix
lcall=lcall/6
print(np.shape(lcall),np.mean(lcall))
sys.exit()
#np.savez("lcdscovr",lcall)
##################################################
N=5
lamA=10**(-2.0) #-6 sparse #-1,-1.5,-2,-2.5,-3,-3.5,-4
lamX=10**(-4.5)
Ntry=50000
epsilon=1.e-12
trytag="D203"
semiNMF=False
regmode="L2-VRDet"
#regmode="L2-VRLD"
#regmode="Dual-L2"
#regmode="Unconstrained"
filename=trytag+"N"+str(N)+regmode+"_A"+str(np.log10(lamA))+"X"+str(np.log10(lamX))

A0,X0=initnmf.init_random(N,npix,lcall)
#X0=X0/10
#A0,X0=initnmf.initpca(N,W,lcall,lamA,mode="Ridge")
#fac=np.sum(lcall)/np.sum(A0)/np.sum(X0)
#A0=A0*fac
A,X,logmetric=runnmf.QP_NMR(regmode,Ntry,lcall,W,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=100,NtryAPGA=1000,eta=1.e-12,endc=-np.inf,Nsave=1000,semiNMF=semiNMF)
np.savez(filename,A,X)

#plt.show()
