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
W,t,lcall=rd.read_dscovr("/home/kawahara/exomap/data/for_HKawahara",4,istart=1)
lcall=lcall*npix
lcall=lcall/6
#print(np.std(lcall)**2*np.shape(lcall)[0]*np.shape(lcall)[1])
#np.savez("lcdscovr",lcall)
##################################################
N=3
lamA=10**(-3.5) #-6 sparse
lamX=1.e1
Ntry=100000
epsilon=1.e-12
trytag="D126x"
regmode="L2-VRDet"
#regmode="L2-VRLD"
#regmode="Dual-L2"
#regmode="Unconstrained"
filename=trytag+regmode+"_A"+str(np.log10(lamA))+"X"+str(np.log10(lamX))

#A0,X0=initnmf.init_random(N,npix,lcall)
#X0=X0/10
A0,X0=initnmf.initpca(N,W,lcall,lamA,mode="Ridge")
np.savez("pcad"+str(np.log10(lamA)),A0,X0)
sys.exit()
#fac=np.sum(lcall)/np.sum(A0)/np.sum(X0)
#A0=A0*fac
A,X,logmetric=runnmf.QP_NMR(regmode,Ntry,lcall,W,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=100,NtryAPGA=1000,eta=1.e-12,endc=-np.inf,Nsave=1000)
np.savez(filename,A,X)

#plt.show()
