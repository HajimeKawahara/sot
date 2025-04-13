"""
Summary
------------------
Core functions for Dynamic SOT

"""

import numpy as np
import scipy
import time 
import sys

def Mean_DYSOT(W,KS,KT,alpha,lc,Pid):
    #mean of the posterior given theta and g
    #Pid: data precision matrix (inverse of the data covariance)    
    Ni,Nj=np.shape(W)
    Kw=alpha*KT*(W@KS@W.T)
    IKw=np.eye(Ni)+Pid@Kw
    Xlc=scipy.linalg.solve(IKw,Pid@lc,assume_a="pos")
    #Aast=alpha*(KT*Xlc)@W@KS
    Aast=alpha*KT@(W.T*Xlc).T@KS
    return Aast

def P_DYSOT(W,KS,KT,alpha,Sigmad):
    # compute P=inv(Pid+KW)
    Kw=alpha*KT*(W@KS@W.T)
    P=scipy.linalg.inv(Sigmad+Kw)
    return P

def PMean_DYSOT(W,KS,KT,alpha,lc,Pid,Sigmad):
    #P and mean of the posterior given theta and g
    #P = (Sigmad + Kw)^-1
    #Pid: data precision matrix (inverse of the data covariance)
    #Sigmad: the data covariance
    Ni,Nj=np.shape(W)
    Kw=alpha*KT*(W@KS@W.T)
    
    IKw=np.eye(Ni)+Pid@Kw
    Xlc=scipy.linalg.solve(IKw,Pid@lc,assume_a="pos")
    #Aast=alpha*(KT*Xlc)@W@KS
    Aast=alpha*KT@(W.T*Xlc).T@KS
    
    P=scipy.linalg.inv(Sigmad+Kw)
    return P,Aast

def Covi_snap(W,KS,KT,alpha,P,i):
    #snapshot covariance
    Bi=alpha*((W@KS).T*KT[i,:]).T
    #Bi=alpha*(W@KS)*np.outer(KT[i,:],np.ones(Nj)) #same but slow
    return alpha*KS*KT[i,i]-Bi.T@P@Bi

def Covj_pixelwise(W,KS,KT,alpha,P,j):
    #pixel-wise covariance
    Cj=alpha*(KT.T*((W@KS)[:,j])).T
    #Cj=alpha*KT*np.outer((W@KS)[:,j],np.ones(Ni)) #same but slow
    return alpha*KS[j,j]*KT-(Cj.T@P@Cj)
