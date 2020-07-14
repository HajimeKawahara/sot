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

up=90
down=10
med=50
Ncreup=np.zeros((Nframe,Nj))
Ncredown=np.zeros((Nframe,Nj))
Ncremed=np.zeros((Nframe,Nj))

for i in range(0,Nframe):
    for n in range(0,Nsample):
        Ncreup[i,n]=np.percentile(randmap[:,i,0,n], up)
        Ncredown[i,n]=np.percentile(randmap[:,i,0,n], down)
        Ncremed[i,n]=np.percentile(randmap[:,i,0,n], med)
        
plotdymap.plotseqmap(Ncreup,range(0,Nframe),"mapup_dscovr","",cmap=plt.cm.pink,Earth=True)
plotdymap.plotseqmap(Ncredown,range(0,Nframe),"mapdown_dscovr","",cmap=plt.cm.pink,Earth=True)
plotdymap.plotseqmap(Ncremed,range(0,Nframe),"mapmed_dscovr","",cmap=plt.cm.pink,Earth=True)

