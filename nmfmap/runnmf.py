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

def QP_MVC_NMF(Ntry,lcall,W,A0,X0,lam,epsilon):
    ### QP_MVC_NMF
    import scipy
    Y=np.copy(lcall)
    NI=np.shape(Y)[0]
    NL=np.shape(Y)[1]
    A=np.copy(A0)
    X=np.copy(X0)
    NK=np.shape(A)[1]
    npix=np.shape(A)[0]
    WTW=np.dot(W.T,W)
    for i in range(0,Ntry):
        if np.mod(i,100)==0:
            ATA = np.dot(A.T,A)
            print(i,np.sum(Y - np.dot(np.dot(W,A),X))+lam*np.linalg.det(ATA))
        G=cp.dot(W,A)
        #Solve Dl = G Xl
        #print("X")
        for l in range(0,NL):
            sol,rnorm=scipy.optimize.nnls(G, Y[:,l])
            X[:,l]=sol
        
        #Solve QPx
        Delta_r = Y - np.dot(np.dot(W,A),X)
        Aprev=np.copy(A)
        #print("A")
        for s in range(0,NK):
            xast2=np.linalg.norm(X[s,:])**2
            Aminus = np.delete(Aprev,obj=s,axis=1)
            #st=time.time()
            #U,S,VT=np.linalg.svd(Aminus)
            #Cs=U[:,2:]
            #CsCsT=np.dot(Cs,Cs.T)
            #ed=time.time()
            #print(ed-st,"sec for Cs")
            #st=time.time()
            ATAinverse=np.linalg.inv(np.dot(Aminus.T,Aminus))
            K=np.eye(npix) - np.dot(np.dot(Aminus,ATAinverse),Aminus.T)
            #ed=time.time()
            #print(ed-st,"sec for K")
            #deltaQ=np.dot(Cs,Cs.T) - K
            #print(np.sum(deltaQ))            
            Delta = Delta_r + np.dot(W,np.outer(A[:,s],X[s,:])) 
            Wcal=xast2*WTW + lam*K
            b = np.dot(np.dot(W.T,Delta),X[s,:])
            sol,rnorm=scipy.optimize.nnls(Wcal, b)
            A[:,s]=sol
        
    return A, X
