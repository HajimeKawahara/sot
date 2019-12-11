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
np.mean(mmap)
lcall=[]
for i in range(0,len(bands)):
    lc=np.dot(W,mmap[:,i])
    ave=np.sum(W,axis=1)
    lcall.append(lc/ave)
lcall=np.array(lcall).T

## NMF multiplicative

normmat=np.diag(1.0/np.sum(lcall,axis=0))
#Y=cp.asarray(lcall)
Y=cp.asarray(np.dot(lcall,normmat))
N=3

## NMF Initialization ============================

#A0=np.random.rand(npix,N)
#X0=np.random.rand(N,np.shape(Y)[1])

import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
pca = PCA(n_components=N)
pca.fit(lcall) #x(k,l)
X0=(np.abs(pca.components_))

iN=np.shape(W)[0]
jN=np.shape(W)[1]
kN=np.shape(X0)[0]
lN=np.shape(X0)[1]

dd=lcall.flatten()
Wd=np.einsum("ij,kl->iljk",W,X0)
Wd=Wd.reshape((iN*lN,jN*kN))
clf = Ridge(alpha=1.e-3)
clf.fit(Wd, dd)
A0=np.abs(clf.coef_.reshape((jN,kN)))
#==================================================


W=cp.asarray(W)
A=cp.asarray(A0)
X=cp.asarray(np.dot(X0,normmat))
epsilon=1.e-7
lam=3.e-4
#lam=1.0
for i in range(0,3000):
    if np.mod(i,10)==0: print(i,cp.sum(Y - cp.dot(cp.dot(W,A),X)))
    ATA = cp.dot(A.T,A)
    Wt = cp.dot(cp.dot(cp.dot(W.T,Y),X.T),ATA)+ epsilon
    Wb = cp.dot(cp.dot(cp.dot(cp.dot(cp.dot(W.T,W),A),X),X.T),ATA) + lam*cp.linalg.det(ATA)*A + epsilon
    #print(np.shape(Wt/Wb),np.shape(A))
    A = A*(Wt/Wb)
    A = cp.dot(A,cp.diag(1/cp.sum(A[:,:],axis=0)))
    Wt = cp.dot(cp.dot(A.T,W.T),Y)+ epsilon
    Wb = cp.dot(cp.dot(cp.dot(cp.dot(A.T,W.T),W),A),X)+ epsilon 
    X = X*(Wt/Wb)

An=cp.asnumpy(A)
Xn=cp.asnumpy(X)
Wn=cp.asnumpy(W)
np.savez("nmftest_10bands_1e0_300000_inipca",An,Xn,Wn,bands)
