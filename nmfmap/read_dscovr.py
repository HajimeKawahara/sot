import numpy as np
import sys
def read_dscovr(dirname,Nreduce):
    #The data was given by Siteng Fan.
    dat=np.load(dirname+"/Matrices.npz")
    W=dat["W"]
    lc2016=np.loadtxt(dirname+"/2016_Data_Land.csv",skiprows=1,delimiter=",")
    lc2017=np.loadtxt(dirname+"/2017_Data_Land.csv",skiprows=1,delimiter=",")
    lc=np.concatenate([lc2016,lc2017])
    t=np.array(lc[:,0])
    lc=np.array(lc[:,1:11])
    if Nreduce > 1:
        lc=lc[::Nreduce,:]
        t=t[::Nreduce]
        W=W[::Nreduce,:]
    return W,t,lc

#W,t,lc=read_dscovr("../../data/for_HKawahara",9)
