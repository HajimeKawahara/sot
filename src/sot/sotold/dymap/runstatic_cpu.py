import numpy as np
import scipy
import time 
import sys

def Mean_STSOT(W,KS,lc,Pid):
    #mean of the posterior given theta and g
    #Pid: data precision matrix (inverse of the data covariance)    
    Kw=(W@KS@W.T)
    Nn=np.shape(Kw)[0]
    IKw=np.eye(Nn)+Pid@Kw        
    Xlc=scipy.linalg.solve(IKw,Pid@lc,assume_a="pos")
    return KS@W.T@Xlc
