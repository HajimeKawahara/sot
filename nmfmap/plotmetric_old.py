def plotall(dat,col):
    plt.plot(dat[1:,0],dat[1:,1],label="Q", color=col,ls="solid")
    plt.plot(dat[1:,0],dat[1:,2],label="Chi2", color=col,ls="dashed")
    plt.plot(dat[1:,0],dat[1:,3],label="A", color=col,ls="dotted")
    plt.plot(dat[1:,0],dat[1:,4],label="X", color=col,ls="dashdot")

import numpy as np
import matplotlib.pyplot as plt


dat100=np.load("npz/metric100.npz",allow_pickle=True)["arr_0"]
dat10=np.load("npz/metric10.npz",allow_pickle=True)["arr_0"]
dat1=np.load("npz/metric1.npz",allow_pickle=True)["arr_0"]


plotall(dat1,"C0")
plotall(dat10,"C1")
plotall(dat100,"C2")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()


