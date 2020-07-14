import numpy as np
import matplotlib.pyplot as plt
import plotdymap
import matplotlib
import healpy as hp
import sys

fontsize=16
matplotlib.rcParams.update({'font.size':fontsize})

dat=np.load("../samples/test531/dymapRBF_529.npz")
Aast_mean=dat["arr_0"]
frames=[78,768,1722] 
plotdymap.plotseqmap(Aast_mean,frames,"map",Earth=True,vmin=-0.75,vmax=0.0)

dat=np.load("../samples/test531/dyranRBF_529.npz")
randmap=dat["arr_0"]
Nj,Nframe,Nn,Nsample=(np.shape(randmap))
Nran=np.zeros((Nframe,Nj))

#for n in range(0,Nsample):
#    hp.mollview(randmap[n,2,0,:])
#x    plt.show()

hp.mollview(np.mean(randmap[:,2,0,:],axis=0),flip="geo")
plt.show()

for n in range(0,Nsample):
    for i in range(0,Nframe):
        Nran[i,n]=randmap[n,i,0,n]
#        Nran[i,n]=randmap[n,i,0,n]/np.mean(randmap[n,i,0,:])


plotdymap.plotseqmap(Nran,range(0,Nframe),"mapran_dscovr","",vmin=-0.75,vmax=0.0,cmap=plt.cm.pink,Earth=True)
#plotdymap.plotseqmap(Nran,range(0,Nframe),"mapran_dscovr","",cmap=plt.cm.pink,Earth=True)
