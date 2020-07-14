import numpy as np
import matplotlib.pyplot as plt
import plotdymap
import matplotlib

fontsize=16
matplotlib.rcParams.update({'font.size':fontsize})

dat=np.load("../samples/test65/dyranRBFspin.npz")
randmap=dat["arr_0"]
Nj,Nframe,Nn,Nsample=(np.shape(randmap))
Nran=np.zeros((Nframe,Nj))

for n in range(0,Nsample):
    for i in range(0,Nframe):
        Nran[i,n]=randmap[n,i,0,n]
plotdymap.plotseqmap(Nran,range(0,Nframe),"mapran","",vmin=0.0,vmax=1.3,cmap=plt.cm.pink)

