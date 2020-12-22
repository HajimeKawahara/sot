"""
Summary
-------------
Static Spin-Orbit Unmixing using a block coordinate descent and GP/QF

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import cupy as cp
import sys
import healpy as hp

from sot.core import io_surface_type 
from sot.core import io_refdata
from sot.core import toymap
from sot.core import mocklc
from sot.nmfmap import initnmf
from sot.core import sepmat 
from sot.dymap import gpkernel 
#from sot.nmfmap import runnmf_cpu as runnmf #CPU version (slow)
from sot.nmfmap import runnmf_gpu as runnmf #GPU version

np.random.seed(34)

## load class map
dataclass=np.load("/home/kawahara/exomap/sot/data/cmap3class.npz")
cmap=dataclass["arr_0"]
npix=len(cmap)
nclass=(len(np.unique(cmap)))
nside=hp.npix2nside(npix)
vals=dataclass["arr_1"]
valexp=dataclass["arr_2"]
print("Nclass=",nclass)

### Set reflectivity
cloud,cloud_ice,snow_fine,snow_granular,snow_med,soil,veg,ice,water,clear_sky\
=io_refdata.read_refdata("/home/kawahara/exomap/sot/data/refdata")

#mean albedo between waves and wavee
#bands=[[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,0.9]]
bands=[[0.4,0.45],[0.45,0.5],[0.5,0.55],[0.55,0.6],[0.6,0.65],[0.65,0.7],[0.7,0.75],[0.75,0.8],[0.8,0.85],[0.85,0.9]]

refsurfaces=[water,soil,veg]
#malbedo=io_surface_type.set_meanalbedo(0.8,0.9,refsurfaces,clear_sky)

mmap,Ainit,Xinit=toymap.make_multiband_map(cmap,refsurfaces,clear_sky,vals,bands)
ave_band=np.mean(np.array(bands),axis=1)
io_surface_type.plot_albedo(veg,soil,cloud,snow_med,water,clear_sky,ave_band,Xinit,valexp)

### Generating Multicolor Lightcurves
inc=45.0/180.0*np.pi
Thetaeq=np.pi/2
zeta=23.4/180.0*np.pi
Pspin=23.9344699/24.0 #Pspin: a sidereal day
wspin=2*np.pi/Pspin 
Porb=365.242190402                                            
worb=2*np.pi/Porb 
N=512
expt=Porb #observation duration 10d
obst=np.linspace(Porb/4,expt+Porb/4,N) 

Thetav=worb*obst
Phiv=np.mod(wspin*obst,2*np.pi)
WI,WV=mocklc.comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]
npix=hp.nside2npix(nside)

lcall=np.dot(np.dot(W,Ainit),Xinit)
noiselevel=0.01
lcall=lcall+noiselevel*np.mean(lcall)*np.random.normal(0.0,1.0,np.shape(lcall))
#np.savez("lcallN"+str(noiselevel),lcall)
#sys.exit()
nside=8
npix=hp.nside2npix(nside)
WI,WV=mocklc.comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]

sep=sepmat.calc_sepmatrix(nside)
gamma=16.5/180.0*np.pi
KS=gpkernel.RBF(sep,gamma)


Nk=3
Ntry=100000
Nsave=10000
epsilon=1.e-12
lamA=10**(-1.0)  #-1---4
lamX=10**(-2.0)   #2, (0,1,3)

## NMF Initialization ============================
A0,X0=initnmf.init_random(Nk,npix,lcall)
#A0,X0=initnmf.initpca(Nk,W,lcall,lamA)
#fac=np.sum(lcall)/np.sum(A0)/np.sum(X0)
#A0=A0*fac

trytag="T215"
#regmode="L2"
regmode="GP-VRDet"
#regmode="L2-VRLD"
#regmode="Dual-L2"
print(KS)

filename=trytag+"_N"+str(Nk)+"_"+regmode+"_A"+str(np.log10(lamA))+"X"+str(np.log10(lamX))
A,X,logmetric=runnmf.GPQP_GNMF(regmode,Ntry,lcall,W,A0,X0,0.1*KS,lamX,epsilon,filename,NtryAPGX=100,NtryAPGA=300,eta=1.e-6,endc=-np.inf,Nsave=Nsave)
np.savez(filename,A,X)




