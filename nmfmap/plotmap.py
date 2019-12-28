import io_surface_type 
import io_refdata
import toymap
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import matplotlib
dat=np.load("npz/ax_dscovr0_0.npz")
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
#bands=[[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,0.9]]#,[0.9,1.0]]
bands=[[0.4,0.45],[0.45,0.5],[0.5,0.55],[0.55,0.6],[0.6,0.65],[0.65,0.7],[0.7,0.75],[0.75,0.8],[0.8,0.85],[0.85,0.9]]
refsurfaces=[water,soil,veg]
malbedo=io_surface_type.set_meanalbedo(0.8,0.9,refsurfaces,clear_sky)
mmap,Ain,Xin=toymap.make_multiband_map(cmap,refsurfaces,clear_sky,vals,bands)
ave_band=np.mean(np.array(bands),axis=1)

cc=plt.cm.viridis
fontsize=18
matplotlib.rcParams.update({'font.size':fontsize})
hp.mollview(A[:,0], title="Component 0",flip="geo",cmap=cc,min=0,max=1)

plt.savefig("C0.pdf", bbox_inches="tight", pad_inches=0.0)

hp.mollview(A[:,1], title="Component 1",flip="geo",cmap=cc,min=0,max=1)
plt.savefig("C1.pdf", bbox_inches="tight", pad_inches=0.0)

hp.mollview(A[:,2], title="Component 2",flip="geo",cmap=cc,min=0,max=1)
plt.savefig("C2.pdf", bbox_inches="tight", pad_inches=0.0)

#hp.mollview(A[:,0]+A[:,1], title="0+1",flip="geo",cmap=plt.cm.jet)


fig= plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)
ax.plot(veg[:,0],veg[:,1],c="black",lw=2,label="vegitation (deciduous)")
ax.plot(soil[:,0],soil[:,1],c="gray",lw=1,label="soil")
#ax.plot(cloud[:,0],cloud[:,1],c="black",ls="dashed",label="cloud (water)")
#ax.plot(snow_med[:,0],snow_med[:,1],c="gray",ls="dashed",label="snow (medium granular)")
ax.plot(water[:,0],water[:,1],c="gray",ls="-.",label="water")
#ax.plot(clear_sky[:,0],clear_sky[:,1],c="gray",ls="dotted",label="clear sky")
#col=["gray","black","black"]
#mal=["X","s","X"]
#for i in range(0,len(valexp)):
#    ax.plot(ave_band,malbedo[i,:],mal[i],label=valexp[i],color=col[i])
plt.xlim(0.4,0.9)

#io_surface_type.plot_albedo(veg,soil,cloud,snow_med,water,clear_sky,ave_band,malbedo,valexp)
fac=0.28
fac0=fac
fac1=fac
fac2=fac


plt.plot(np.median(bands,axis=1),X[0,:]*fac0,"o",label="Component 0",color="C0")
plt.plot(np.median(bands,axis=1),X[1,:]*fac1,"s",label="Component 1",color="C1")
plt.plot(np.median(bands,axis=1),X[2,:]*fac2,"^",label="Component 2",color="C2")
#plt.plot(np.median(bands,axis=1),(X[0,:]+X[1,:])*fac3,"^",label="Component 0+1",color="C1")

plt.plot(np.median(bands,axis=1),X[0,:]*fac0,color="C0")
plt.plot(np.median(bands,axis=1),X[1,:]*fac1,color="C1")
plt.plot(np.median(bands,axis=1),X[2,:]*fac2,color="C2")
#plt.plot(np.median(bands,axis=1),(X[0,:]+X[1,:])*fac3,color="C1")

plt.tick_params(labelsize=16)
plt.ylabel("Reflection Spectra",fontsize=16)
plt.xlabel("wavelength [micron]",fontsize=16)
plt.legend(fontsize=13)

plt.savefig("ref.pdf", bbox_inches="tight", pad_inches=0.0)

A[(A>0.333333)*(A<0.333334)]=0
Aclass=np.argmax(A,axis=1)
print(Aclass)
Aclass[Aclass==0]=100
Aclass[Aclass==1]=70
Aclass[Aclass==2]=0

hp.mollview(Aclass, title="Retrieved",flip="geo",cmap=plt.cm.Greys,max=100)
plt.savefig("retrieved.pdf", bbox_inches="tight", pad_inches=0.0)


dataclass=np.load("/home/kawahara/exomap/sot/data/cmap3class.npz")
cmapans=dataclass["arr_0"]
hp.mollview(cmapans, title="Input",flip="geo",cmap=plt.cm.Greys_r)
plt.savefig("input.pdf", bbox_inches="tight", pad_inches=0.0)

plt.show()
