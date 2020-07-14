import numpy as np
import matplotlib.pyplot as plt
import plotdymap
import matplotlib
import healpy as hp
fontsize=16
matplotlib.rcParams.update({'font.size':fontsize})

dat=np.load("../samples/test65/dyranRBFspin.npz")
randmap=dat["arr_0"]
Nj,Nframe,Nn,Nsample=(np.shape(randmap))
up=95
down=5
med=50
Ncreup=np.zeros((Nframe,Nj))
Ncredown=np.zeros((Nframe,Nj))
Ncremed=np.zeros((Nframe,Nj))

for i in range(0,Nframe):
    for n in range(0,Nsample):
        Ncreup[i,n]=np.percentile(randmap[:,i,0,n], up)
        Ncredown[i,n]=np.percentile(randmap[:,i,0,n], down)
        Ncremed[i,n]=np.percentile(randmap[:,i,0,n], med)
        
plotdymap.plotseqmap(Ncreup,range(0,Nframe),"mapup","",cmap=plt.cm.pink)
plotdymap.plotseqmap(Ncredown,range(0,Nframe),"mapdown","",cmap=plt.cm.pink)
plotdymap.plotseqmap(Ncremed,range(0,Nframe),"mapmed","",cmap=plt.cm.pink)

