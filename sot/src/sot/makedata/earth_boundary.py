#!/usr/bin/env python
import numpy as np
import healpy as hp
import pylab 
import matplotlib.pyplot as plt
import matplotlib

dataclass=np.load("/home/kawahara/exomap/sot/data/cmap3class.npz")
cmap=dataclass["arr_0"]
cmap[cmap>0.1]=1.0
#hp.mollview(cmap, title="",flip="geo",cmap=plt.cm.bone)#,min=0.0,max=1.0)
#plt.show()
npix=len(cmap)
nside=hp.npix2nside(npix)
cmapx=np.copy(cmap)
for i in range(0,npix):
    if cmap[i]>0:
        nei=hp.pixelfunc.get_interp_weights(nside,i)
        val4=cmap[nei[0]]
        if np.sum(val4)==4:
            cmapx[i]=0

zmap=np.zeros(len(cmap))
for i in range(0,npix):
    if cmapx[i]>0:
        nni=hp.pixelfunc.get_all_neighbours(nside,i)
        val8=cmap[nni]
        if np.sum(val8)>0:
            zmap[i]=1

#hp.mollview(zmap, title="",flip="geo",cmap=plt.cm.bone)#,min=0.0,max=1.0)
#plt.show()

mask=zmap==1
pix=np.array(range(npix))
np.savez("earth_boundary",nside,pix[mask])

#test
dat=np.load("earth_boundary.npz")
nside=dat["arr_0"]
nbound=dat["arr_1"]

zzmap=np.zeros(len(cmap))
hp.mollview(zzmap, title="",flip="geo",cmap=plt.cm.bone)#,min=0.0,max=1.0)
theta,phi=hp.pixelfunc.pix2ang(nside,nbound)

hp.projplot(theta, phi,".",c="white",alpha=0.5) 
plt.show()


