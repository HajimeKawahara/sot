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
malbedo=io_surface_type.set_meanalbedo(0.8,0.9,refsurfaces,clear_sky)
mmap,malbedo=toymap.make_multiband_map(cmap,refsurfaces,clear_sky,vals,bands)
ave_band=np.mean(np.array(bands),axis=1)
io_surface_type.plot_albedo(veg,soil,cloud,snow_med,water,clear_sky,ave_band,malbedo,valexp)
#plt.show()

### Generating Multicolor Lightcurves
#ephemeris setting
inc=0.0
Thetaeq=np.pi/2
#zeta=23.4/180.0*np.pi
zeta=90.0/180.0*np.pi 
Pspin=23.9344699/24.0 #Pspin: a sidereal day
wspin=2*np.pi/Pspin 
#Porb=365.242190402                                            
Porb=30.0
worb=2*np.pi/Porb 
N=1024
expt=Porb #observation duration 10d
obst=np.linspace(Porb/4,expt+Porb/4,N) 

Thetav=worb*obst
Phiv=np.mod(wspin*obst,2*np.pi)
WI,WV=mocklc.comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]

npix=hp.nside2npix(nside)
lcall=[]
for i in range(0,len(bands)):
    lc=np.dot(W,mmap[:,i])
#    ave=np.sum(W,axis=1)
#    lcall.append(lc/ave)
    lcall.append(lc)
lcall=np.array(lcall).T

#normmat=np.diag(1.0/np.sum(lcall,axis=0))
#Y=cp.asarray(np.dot(lcall,normmat))
N=3


## NMF Initialization ============================
#A0,X0=init_random(N,npix,Y)
A0,X0=initnmf.initpca(N,W,lcall)

#lam=3.e-4
#lam=3.e-5

#X=cp.asarray(np.dot(X0,normmat))

#print(np.shape(mmap),np.shape(cmap),np.shape(malbedo))
#cTc=np.dot(cmap.T,cmap)
#Qtrue=np.sum(lcall - np.dot(Wn,mmap))+lam*np.linalg.det(cTc)
#print(Qtrue)
#sys.exit()

Ntry=300
lam=1.0
epsilon=1.e-9


#A,X=runnmf.NG_MVC_NMF(Ntry,lcall,W,A0,X0,lam,epsilon)

import scipy
Y=np.copy(lcall)
NI=np.shape(Y)[0]
NL=np.shape(Y)[1]
A=np.copy(A0)
X=np.copy(X0)
NK=np.shape(A)[1]

for i in range(0,Ntry):
    ATA = cp.dot(A.T,A)
    if np.mod(i,10)==0: print(i,np.sum(Y - np.dot(np.dot(W,A),X))+lam*np.linalg.det(ATA))
    G=cp.dot(W,A)
    #Solve Dl = G Xl
    for l in range(0,NL):
        X[:,l]=scipy.optimize.nnls(G, Y[:,l])


    #Solve
    for k in range(0,NK):
        h=np.lingalg.norm(X[k,:])
        Wi=

        matrixA = h*h*np.eye(NI)+
