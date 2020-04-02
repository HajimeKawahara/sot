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
from sklearn.decomposition import NMF
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

#compute lambert beta function
Aunif=np.ones(np.shape(Ainit)[0])
lambertf=np.dot(W,Aunif)
np.savez("lambert",lambertf)

lcall=np.dot(np.dot(W,Ainit),Xinit)
lcall=(lcall.T/lambertf).T

noiselevel=0.01
lcall=lcall+noiselevel*np.mean(lcall)*np.random.normal(0.0,1.0,np.shape(lcall))


#fig=plt.figure()
#plt.plot(lambertf)
#plt.plot(lcall[:,3])
#plt.show()
#sys.exit()

Nk=3
Ntry=10000
epsilon=1.e-12

lamA=0.0  #-1---4
lamX=10**(-1)

## NMF Initialization ============================
# USE NMF (no regularization) in scikit-learn
model = NMF(n_components=Nk, init='random', random_state=0)
A0 = model.fit_transform(lcall)
X0 = model.components_
np.savez("directNMF",A0,X0)

trytag="LC401"
#regmode="L2"
regmode="L2-VRDet"
#regmode="L2-VRLD"
#regmode="Dual-L2"

filename=trytag+"_N"+str(Nk)+"_"+regmode+"_A"+str(np.log10(lamA))+"X"+str(np.log10(lamX))
#A,X,logmetric=runnmf.QP_NMF(regmode,Ntry,lcall,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=1000,NtryAPGA=1,eta=1.e-16,endc=0.0,Lipfaca=1.e-3,Lipfacx=1.e-3)
A,X,logmetric=runnmf.QP_NMF(regmode,Ntry,lcall,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=1000,NtryAPGA=1,eta=1.e-16,endc=0.0,Lipfaca=1.e-2,Lipfacx=1.0)

print("mean residual=",np.sqrt(np.sum(lcall-A@X)**2)/np.shape(lcall)[0])


AX=A@X
res=lcall-AX
fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(lcall[:,0])
plt.plot(AX[:,0],".")
ax=fig.add_subplot(212)
plt.plot(res[:,0])
plt.show()

fig=plt.figure()
ax=fig.add_subplot(211)
for k in range(0,Nk):
    plt.plot(X[k,:])
plt.show()


np.savez(filename,A,X)



