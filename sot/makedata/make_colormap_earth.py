import numpy as np
import io_surface_type 
import io_refdata
import toymap
import mocklc
import matplotlib.pyplot as plt
import healpy as hp
import sys

## load class map
dataclass=np.load("../../data/cmap3class.npz")
cmap=dataclass["arr_0"]
npix=len(cmap)
nclass=(len(np.unique(cmap)))
nside=hp.npix2nside(npix)
vals=dataclass["arr_1"]
valexp=dataclass["arr_2"]
print("Nclass=",nclass)
# ## Set reflectivity

cloud,cloud_ice,snow_fine,snow_granular,snow_med,soil,veg,ice,water,clear_sky\
=io_refdata.read_refdata("../../../data/refdata")
#mean albedo between waves and wavee
print(io_refdata.get_meanalbedo(veg,0.8,0.9),io_refdata.get_meanalbedo(veg,0.4,0.5))


bands=[[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8]]
#bands=[[0.4,0.5],[0.6,0.7],[0.8,0.9]]
#bands=[[0.4,0.5],[0.6,0.7],[0.8,0.9],[1.0,1.1]]
#bands=[[0.4,0.45],[0.45,0.5],[0.5,0.55],[0.55,0.6],[0.6,0.65],\
#       [0.65,0.7],[0.7,0.75],[0.75,0.8],[0.8,0.85],[0.85,0.9]]
if nclass==4:
    refsurfaces=[water,soil,snow_med,veg]
elif nclass==3:
    refsurfaces=[water,soil,veg]
else:
    print("Nclass should be 3 or 4 currently")
    sys.exit()
    
malbedo=io_surface_type.set_meanalbedo(0.8,0.9,refsurfaces,clear_sky)
mmap,malbedo=toymap.make_multiband_map(cmap,refsurfaces,clear_sky,vals,bands)
ave_band=np.mean(np.array(bands),axis=1)
io_surface_type.plot_albedo(veg,soil,cloud,snow_med,water,clear_sky,ave_band,malbedo,valexp)

# ## Generating Multicolor Lightcurves

#ephemeris setting
inc=np.pi/2.0
Thetaeq=np.pi/2
zeta=23.4/180.0*np.pi                                                                                        
Pspin=23.9344699/24.0 #Pspin: a sidereal day
wspin=2*np.pi/Pspin 
Porb=365.242190402                                            
#Porb=10.0
worb=2*np.pi/Porb 
N=1024
expt=Porb #observation duration 10d
obst=np.linspace(Porb/4,expt+Porb/4,N) 

Thetav=worb*obst
Phiv=np.mod(wspin*obst,2*np.pi)
WI,WV=mocklc.comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv)
W=WV[:,:]*WI[:,:]


# +
### Dont forget pre-whitening !!!! 
# -

import healpy as hp
npix=hp.nside2npix(nside)
np.mean(mmap)
lcall=[]
for i in range(0,len(bands)):
    lc=np.dot(W,mmap[:,i])
    ave=np.sum(W,axis=1)
    #sigma=np.std(lc)*0.05
    #noise=np.random.normal(0.0,sigma,len(lc))
    #lc=lc+noise
    #prewhitening
    #lc=(lc-np.mean(lc))
    #lc=lc/np.std(lc)
    #print(np.mean(lc),np.std(lc))
    #lcall.append(lc)
    lcall.append(lc/ave)
lcall=np.array(lcall).T
print(np.shape(lcall))

plt.close()
for i in range(0,len(bands)):
    plt.plot(lcall[:,i],".")
plt.show()

np.savez("lcall4",lcall)
