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
W,t,lcall=rd.read_dscovr("/home/kawahara/exomap/data/for_HKawahara",8)
lcall=lcall
noiselevel=0.0001
lcall=lcall+noiselevel*np.mean(lcall)*np.random.normal(0.0,1.0)
##################################################

N=3
lamA=1.e-2
lamX=1.e0
Ntry=100000
epsilon=1.e-12
#regmode="L2-VRDet"
regmode="L2-VRLD"
#regmode="Unconstrained"
filename="xDSCOVR"+regmode+"AX_a"+str(np.log10(lamA))+"x"+str(np.log10(lamX))+"_try"+str(Ntry)

A0,X0=initnmf.init_random(N,npix,lcall)

A,X,logmetric=runnmf.QP_NMR(regmode,Ntry,lcall,W,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=100,NtryAPGA=1000,eta=1.e-12)
np.savez(filename,A,X)

#plt.show()
