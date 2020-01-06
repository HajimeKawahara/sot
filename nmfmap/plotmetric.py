import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

ini=0
for i in range(0,3):
    dat=np.array(np.load("metricVRx0_"+str(i)+".npz")["arr_0"])
    dat=np.array(cp.asnumpy(dat),dtype=np.float)
    plt.plot(dat[:,0]+ini,dat[:,1],label="Q",color="C0")
    plt.plot(dat[:,0]+ini,dat[:,2],label="chi2",color="C1")
    plt.plot(dat[:,0]+ini,dat[:,3],label="A",color="C2")
    plt.plot(dat[:,0]+ini,dat[:,4],label="X",color="C3")
    ini=dat[-1,0]+ini
    if i==0:
        plt.legend()
    
plt.xscale("log")
plt.yscale("log")
plt.savefig("metric.png")
