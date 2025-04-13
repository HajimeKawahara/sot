import numpy as np
import sys
import datetime
from astropy.time import Time
import pandas as pd
import tqdm

def read_dscovr(dirname,Nreduce,Nfin=None,istart=3,timeobj=False):
    #The data was given by Siteng Fan.
    dat=np.load(dirname+"/Matrices.npz")
    W=dat["W"]
    t=pd.read_csv(dirname+"/2016_Data_Land.csv",delimiter=",",dtype='object',usecols=((0,)))
    lc2016=np.loadtxt(dirname+"/2016_Data_Land.csv",skiprows=1,delimiter=",")

    lab=[]
    tframe=[]
    for dstr in tqdm.tqdm(t["Time"]):
        date="2016"+str(dstr)
        tdate = datetime.datetime.strptime(date, '%Y%m%d%H%M%S')
        tframe.append(float(Time(tdate).jd))
        lab.append(tdate)
        
    t=pd.read_csv(dirname+"/2017_Data_Land.csv",delimiter=",",dtype='object',usecols=((0,)))
    lc2017=np.loadtxt(dirname+"/2017_Data_Land.csv",skiprows=1,delimiter=",")
    for dstr in tqdm.tqdm(t["Time"]):
        date="2017"+str(dstr)
        tdate = datetime.datetime.strptime(date, '%Y%m%d%H%M%S')
        tframe.append(float(Time(tdate).jd))
        lab.append(tdate)
        
    tframe=np.array(tframe)
    lc=np.concatenate([lc2016,lc2017])
    lc=np.array(lc[:,1+istart:11])
    if Nreduce > 1:
        lc=lc[::Nreduce,:]
        tframe=tframe[::Nreduce]
        W=W[::Nreduce,:]
        if timeobj:
            lab=lab[::Nreduce]
            
    if Nfin is not None:
        lc=lc[0:Nfin]
        tframe=tframe[0:Nfin]
        W=W[0:Nfin,:]
        if timeobj:
            lab=lab[0:Nfin]

    #zero point
    tdate=datetime.datetime.strptime("20160101000000", '%Y%m%d%H%M%S')
    t0=float(Time(tdate).jd)
    tframe=tframe-t0

    if timeobj:
        return W,tframe,lc,np.array(lab)
    else:        
        return W,tframe,lc


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib
    fontsize=16
    matplotlib.rcParams.update({'font.size':fontsize})

    W,t,lc,lab=read_dscovr("../../data/for_HKawahara",1,timeobj=True)
    print(lab)

    fig=plt.figure()
    ax=fig.add_subplot(211)
    plt.plot(t,lc[0:,1],".",color="black")
    plt.plot(t,lc[0:,1],alpha=0.5,color="gray")
    plt.ylabel("0.44 $\\mu$m")

    plt.xlim(0,30)
    plt.ylim(0.28,0.33)

    ax=fig.add_subplot(212)
    plt.plot(t,lc[0:,6],".",color="black")
    plt.plot(t,lc[0:,6],alpha=0.5,color="gray")
    plt.ylabel("0.78 $\\mu$m")
    plt.xlabel("day")

    plt.xlim(0,30)
    plt.ylim(0.19,0.28)
    plt.savefig("lcdscvr.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()
