import cupy as cp
import numpy as np

def NG_MVC_NMF(Ntry,lcall,Win,A0,X0,lam,epsilon):
    Y=cp.asarray(lcall)
    W=cp.asarray(Win)
    A=cp.asarray(A0)
    X=cp.asarray(X0)

    for i in range(0,Ntry):
        ATA = cp.dot(A.T,A)
        if np.mod(i,10)==0: print(i,cp.sum(Y - cp.dot(cp.dot(W,A),X))+lam*cp.linalg.det(ATA))
        
        Wt = cp.dot(cp.dot(cp.dot(W.T,Y),X.T),ATA)+ epsilon
        Wb = cp.dot(cp.dot(cp.dot(cp.dot(cp.dot(W.T,W),A),X),X.T),ATA) + lam*cp.linalg.det(ATA)*A + epsilon
        #print(np.shape(Wt/Wb),np.shape(A))
        A = A*(Wt/Wb)
        A = cp.dot(A,cp.diag(1/cp.sum(A[:,:],axis=0)))
        Wt = cp.dot(cp.dot(A.T,W.T),Y)+ epsilon
        Wb = cp.dot(cp.dot(cp.dot(cp.dot(A.T,W.T),W),A),X)+ epsilon 
        X = X*(Wt/Wb)

    A=cp.asnumpy(A)
    X=cp.asnumpy(X)
        
    return A, X
