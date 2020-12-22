import numpy as np

def residual(d,W,M):
    rL=d - np.sum(W*M,axis=1)
    return rL

def dLdM(W,rL):
    return - (W.T*rL.T).T

def dRMdM(M,invKS,invKT,alpha):
    return invKT@M@invKS/alpha
