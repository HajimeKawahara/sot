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
#import runnmf_cpu as runnmf #CPU version (slow)
import runnmf_gpu as runnmf #GPU version

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
N=1024
expt=Porb #observation duration 10d
obst=np.linspace(Porb/4,expt+Porb/4,N) 

Thetav=worb*obst
Phiv=np.mod(wspin*obst,2*np.pi)
WI,WV=mocklc.comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]
npix=hp.nside2npix(nside)

lcall=np.dot(np.dot(W,Ainit),Xinit)
noiselevel=0.0001
lcall=lcall+noiselevel*np.mean(lcall)*np.random.normal(0.0,1.0)
#lcall= np.dot(np.diag(1/np.sum(lcall[:,:],axis=1)),lcall)
#np.savez("lcall",lcall)
#sys.exit()

nside=16
npix=hp.nside2npix(nside)
WI,WV=mocklc.comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]
N=3
Ntry=100000
epsilon=1.e-6
lamA=1.e-2
lamX=1.e2

## NMF Initialization ============================
A0,X0=initnmf.init_random(N,npix,lcall)
#A0,X0=initnmf.initpca(N,W,lcall,lamA)
#Ntryini=10000
#Initialization by Multiplicative Update
#A,X,logmetric=runnmf.MP_L2_NMF(Ntryini,lcall,W,A0,X0,lamA,0.0,epsilon)
#A0,X0=A,X
#np.savez("init_uncMP",A,X)
## ===============================================
#dat=np.load("init_uncMP.npz")
#A0=dat["arr_0"]
#X0=dat["arr_1"]

#regmode="L2-VRDet"
regmode="L2-VRLD"

filename=regmode+"AX_a"+str(np.log10(lamA))+"x"+str(np.log10(lamX))+"_try"+str(Ntry)

A,X,logmetric=runnmf.QP_NMR(regmode,Ntry,lcall,W,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=100,NtryAPGA=300,eta=1.e-6)
np.savez(filename,A,X)

