# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

import numpy as np
import pylab
import matplotlib.pyplot as plt
import healpy as hp
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import io_surface_type 
import io_refdata
import toymap
import mocklc
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

# ## Set Spectral Regions on a Global Map

cmap=io_surface_type.read_classification("../../data/global_2008_5min.asc")

fig = plt.figure(figsize=(10,5))
ax=fig.add_subplot(111)
a=ax.imshow(cmap,cmap="tab20b")
plt.colorbar(a)
plt.title("MODIS CLASSIFICATION MAP 2008")
plt.show()

cmap=io_surface_type.read_classification("../../data/global_2008_5min.asc")
cmap,vals,valexp=io_surface_type.merge_to_4classes(cmap)
cmap,nside=io_surface_type.copy_to_healpix(cmap,nside=64)

hp.mollview(cmap, title="test",flip="geo",cmap=plt.cm.gray)
hp.graticule(color="white")

# ## Set reflectivity

cloud,cloud_ice,snow_fine,snow_granular,snow_med,soil,veg,ice,water,clear_sky\
=io_refdata.read_refdata("../../data/refdata")

fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(veg[:,0],veg[:,1],c="black",lw=2,label="vegitation (deciduous)")
ax.plot(soil[:,0],soil[:,1],c="gray",lw=1,label="soil")
ax.plot(cloud[:,0],cloud[:,1],c="black",ls="dashed",label="cloud (water)")
ax.plot(snow_med[:,0],snow_med[:,1],c="gray",ls="dashed",label="snow (medium granular)")
ax.plot(water[:,0],water[:,1],c="gray",ls="-.",label="water")
ax.plot(clear_sky[:,0],clear_sky[:,1],c="gray",ls="dotted",label="clear sky")
plt.xlim(0.3,2.7)
plt.ylim(0,1.1)
plt.legend(loc="upper right",prop={'size':11},frameon=False)
plt.tick_params(labelsize=14)
plt.ylabel("reflectivity",fontsize=16)
plt.xlabel("wavelength [micron]",fontsize=14)
plt.savefig("reflectivity.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()

#mean albedo between waves and wavee
print(io_refdata.get_meanalbedo(veg,0.8,0.9),io_refdata.get_meanalbedo(veg,0.4,0.5))

valexp

# ## Mixing Spectral Component

#bands=[[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,0.9]]
#bands=[[0.4,0.5],[0.6,0.7],[0.8,0.9]]
bands=[[0.4,0.5],[0.6,0.7],[0.8,0.9],[1.0,1.1]]
#bands=[[0.4,0.45],[0.45,0.5],[0.5,0.55],[0.55,0.6],[0.6,0.65],\
#       [0.65,0.7],[0.7,0.75],[0.75,0.8],[0.8,0.85],[0.85,0.9]]
refsurfaces=[water,soil,snow_med,veg]
malbedo=io_surface_type.set_meanalbedo(0.8,0.9,refsurfaces,clear_sky)

mmap,malbedo=toymap.make_multiband_map(cmap,refsurfaces,clear_sky,vals,bands)

ave_band=np.mean(np.array(bands),axis=1)

fig= plt.figure()
ax = fig.add_subplot(111)
ax.plot(veg[:,0],veg[:,1],c="black",lw=2,label="vegitation (deciduous)")
ax.plot(soil[:,0],soil[:,1],c="gray",lw=1,label="soil")
ax.plot(cloud[:,0],cloud[:,1],c="black",ls="dashed",label="cloud (water)")
ax.plot(snow_med[:,0],snow_med[:,1],c="gray",ls="dashed",label="snow (medium granular)")
ax.plot(water[:,0],water[:,1],c="gray",ls="-.",label="water")
ax.plot(clear_sky[:,0],clear_sky[:,1],c="gray",ls="dotted",label="clear sky")
for i in range(0,len(valexp)):
    ax.plot(ave_band,malbedo[i,:],"+",label=valexp[i])
plt.xlim(0.4,1.5)
plt.legend(bbox_to_anchor=(1.1, 0.3))
plt.show()

hp.mollview(mmap[:,2], title="",flip="geo",cmap=plt.cm.jet)
hp.graticule(color="white")

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
expt=3 #observation duration 10d
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
np.shape(lcall)

# +
ccmap=io_surface_type.read_classification("../../data/global_2008_5min.asc")
ccmap,vals,valexp=io_surface_type.merge_to_4classes(ccmap)
cx=["gray","black","black"]
lsx=["solid","solid","dashed"]
fig= plt.figure(figsize=(7,7))
ax = fig.add_subplot(212)
for i in range(0,3):
    ax.plot(obst-obst[0],lcall[:,i],lw=1,label=str(bands[i][0])+"-"+str(bands[i][1])+" nm",\
            color=cx[i],ls=lsx[i])
#plt.xlim(0.3,2.7)
#plt.ylim(0,1.1)
plt.legend(loc="upper right",prop={'size':11})#,frameon=False)
plt.tick_params(labelsize=18)
plt.ylabel("albedo",fontsize=18)
plt.xlabel("time [d]",fontsize=18)


ax = fig.add_subplot(211)
plt.title("cloudless Earth",fontsize=18)
plt.imshow(ccmap,cmap="gray")
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
plt.savefig("sotlc.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()
# -

ica = FastICA(n_components=len(bands))
S_ = ica.fit_transform(lcall) 
for i in range(0,np.shape(S_)[1]):
    S_[:,i]=(S_[:,i]-np.mean(S_[:,i]))
    S_[:,i]=S_[:,i]/np.std(S_[:,i])

ecmap=toymap.make_ecmap(cmap,vals)
eclcall=[]
for i in range(0,len(vals)):
    eclc=np.dot(W,ecmap[:,i])
    eclc=(eclc-np.mean(eclc))
    eclc=eclc/np.std(eclc)
    eclcall.append(eclc)
eclcall=np.array(eclcall).T
np.shape(eclcall)

# +
ss=0
aa=3
fig=plt.figure(figsize=(14,10))
ax=fig.add_subplot(421)
ax.plot(obst,-S_[:,0],lw=1)
plt.xlim(ss,aa)
ax=fig.add_subplot(423)
ax.plot(obst,S_[:,1],lw=1)
plt.xlim(ss,aa)
ax=fig.add_subplot(425)
ax.plot(obst,S_[:,2],lw=1)
plt.xlim(ss,aa)
ax=fig.add_subplot(427)
ax.plot(obst,-S_[:,3],lw=1)
plt.xlim(ss,aa)

ax=fig.add_subplot(422)
ax.plot(obst,eclcall[:,0],lw=1,label=valexp[0])
ax.plot(obst,S_[:,1],lw=1)
plt.legend()
plt.xlim(ss,aa)

ax=fig.add_subplot(424)
ax.plot(obst,eclcall[:,2],lw=1,label=valexp[2])
ax.plot(obst,S_[:,2],lw=1)
plt.legend()
plt.xlim(ss,aa)

ax=fig.add_subplot(426)
ax.plot(obst,eclcall[:,1],lw=1,label=valexp[1])
ax.plot(obst,-S_[:,1]-S_[:,0],lw=1,label="0 + 3")
plt.legend()
plt.xlim(ss,aa)

ax=fig.add_subplot(428)
ax.plot(obst,eclcall[:,3],lw=1,label=valexp[3])
#ax.plot(obst,S_[:,2],lw=1)
ax.plot(obst,-S_[:,1]+S_[:,0],lw=1,label="0 - 3")
plt.legend()
plt.xlim(ss,aa)
plt.show()

# +
ss=0
aa=3
fig=plt.figure(figsize=(14,10))
ax=fig.add_subplot(421)
ax.plot(obst,-S_[:,0],lw=1)
plt.xlim(ss,aa)
ax=fig.add_subplot(423)
ax.plot(obst,S_[:,1],lw=1)
plt.xlim(ss,aa)
ax=fig.add_subplot(425)
ax.plot(obst,S_[:,2],lw=1)
plt.xlim(ss,aa)
ax=fig.add_subplot(427)
ax.plot(obst,-S_[:,3],lw=1)
plt.xlim(ss,aa)

ax=fig.add_subplot(422)
ax.plot(obst,eclcall[:,0],lw=1,label=valexp[0])
ax.plot(obst,-S_[:,0],lw=1)
plt.legend()
plt.xlim(ss,aa)

ax=fig.add_subplot(424)
ax.plot(obst,eclcall[:,2],lw=1,label=valexp[2])
ax.plot(obst,S_[:,2],lw=1)
plt.legend()
plt.xlim(ss,aa)

ax=fig.add_subplot(426)
ax.plot(obst,eclcall[:,1],lw=1,label=valexp[1])
ax.plot(obst,S_[:,0]+S_[:,3],lw=1,label="0 + 3")
plt.legend()
plt.xlim(ss,aa)

ax=fig.add_subplot(428)
ax.plot(obst,eclcall[:,3],lw=1,label=valexp[3])
#ax.plot(obst,S_[:,2],lw=1)
ax.plot(obst,S_[:,0]-S_[:,3],lw=1,label="0 - 3")
plt.legend()
plt.xlim(ss,aa)
plt.show()

# +
#陸の裏が海っていう拘束条件がついてるから、oceanを1として　1+3，1-3みたいな感じで分離される
#例として２値分類を考えてみれば良いかも。
# -

A_ = ica.mixing_ 
Srecov= np.dot(S_, A_.T) + ica.mean_

fig= plt.figure()
ax = fig.add_subplot(121)
for i in range(0,len(A_)):
    ax.plot(ave_band,A_[:,i]/np.std(A_[:,i]-np.mean(A_[:,i])),".")
ax = fig.add_subplot(122)
for i in range(0,len(valexp)):
    ax.plot(ave_band,malbedo[i,:]/np.std(malbedo[i,:]-np.mean(malbedo[i,:])),"+",label=valexp[i])
plt.legend()
plt.show()

np.shape(Srecov)

fig= plt.figure()
ax = fig.add_subplot(111)
for i in range(0,len(A_)):
    ax.plot(obst,Srecov[:,i])
plt.show()


