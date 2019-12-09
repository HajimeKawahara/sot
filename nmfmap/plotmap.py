import io_surface_type 
import io_refdata
import toymap
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

dat=np.load("nmftest.npz")
A=dat["arr_0"]
X=dat["arr_1"]


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
bands=[[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,0.9],[0.9,1.0]]
refsurfaces=[water,soil,veg]
malbedo=io_surface_type.set_meanalbedo(0.8,0.9,refsurfaces,clear_sky)
mmap,malbedo=toymap.make_multiband_map(cmap,refsurfaces,clear_sky,vals,bands)
ave_band=np.mean(np.array(bands),axis=1)
io_surface_type.plot_albedo(veg,soil,cloud,snow_med,water,clear_sky,ave_band,malbedo,valexp)
fac0=0.3
fac1=0.06
fac2=0.4

plt.plot(np.median(bands,axis=1),X[0,:]*fac0,label="0")
plt.plot(np.median(bands,axis=1),X[1,:]*fac1,label="1")
plt.plot(np.median(bands,axis=1),X[2,:]*fac2,label="2")
plt.legend()
plt.show()



hp.mollview(A[:,0], title="",flip="geo",cmap=plt.cm.jet)
plt.show()
hp.mollview(A[:,1], title="",flip="geo",cmap=plt.cm.jet)
plt.show()
hp.mollview(A[:,2], title="",flip="geo",cmap=plt.cm.jet)
plt.show()

