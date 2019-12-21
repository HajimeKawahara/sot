import cupy as cp
import numpy as np


def NG_L2MVC_NMF(Ntry,lcall,Win,A0,X0,lam,epsilon):
    Y=cp.asarray(lcall)
    W=cp.asarray(Win)
    A=cp.asarray(A0)
    X=cp.asarray(X0)
    logmetric=[]
    jj=0
    for i in range(0,Ntry):
        AA=np.sum(A*A)
        ATA = cp.dot(A.T,A)

        #----------------------------------------------------
        if np.mod(i,1000)==0:
            jj=jj+1
            chi2=cp.sum((Y - cp.dot(cp.dot(W,A),X))**2)
            metric=[i,cp.asnumpy(chi2+lam*AA),cp.asnumpy(chi2),lam*AA]
            logmetric.append(metric)
            #            print(metric,np.sum(A),np.sum(X))
            import terminalplot
            Xn=cp.asnumpy(X)
            bandl=np.array(range(0,len(Xn[0,:])))
            terminalplot.plot(list(bandl),list(Xn[np.mod(jj,3),:]))

            if np.mod(i,10000)==0:
                LogNMF(i,cp.asnumpy(A),cp.asnumpy(X))
        #----------------------------------------------------
            
        Wt = cp.dot(cp.dot(cp.dot(W.T,Y),X.T),ATA)+ epsilon
        Wb = cp.dot(cp.dot(cp.dot(cp.dot(cp.dot(W.T,W),A),X),X.T),ATA) + lam*cp.linalg.det(ATA)*A + lam*2*cp.dot(A,ATA) + epsilon
        A = A*(Wt/Wb)

        A = A*(Wt/Wb)
        #A = cp.dot(cp.diag(1/cp.sum(A[:,:],axis=1)),A)
        A = cp.dot(cp.diag(1/cp.sum(A[:,:],axis=1)),A)
        
        Wt = cp.dot(cp.dot(A.T,W.T),Y)+ epsilon
        Wb = cp.dot(cp.dot(cp.dot(cp.dot(A.T,W.T),W),A),X)+ epsilon 
        X = X*(Wt/Wb)
        #X = cp.dot(cp.diag(1/cp.sum(X[:,:],axis=1)),X)
        #X = cp.dot(X,cp.diag(1/cp.sum(X[:,:],axis=0)))
        
    A=cp.asnumpy(A)
    X=cp.asnumpy(X)
    #----------------------------------------------------
    LogMetricPlot(logmetric)
    #----------------------------------------------------

    return A, X


def L2_NMF(Ntry,lcall,Win,A0,X0,lam,epsilon):
    Y=cp.asarray(lcall)
    W=cp.asarray(Win)
    A=cp.asarray(A0)
    X=cp.asarray(X0)
    logmetric=[]
    jj=0
    for i in range(0,Ntry):
        AA=np.sum(A*A)
        #----------------------------------------------------
        if np.mod(i,1000)==0:
            jj=jj+1
            chi2=cp.sum((Y - cp.dot(cp.dot(W,A),X))**2)
            metric=[i,cp.asnumpy(chi2+lam*AA),cp.asnumpy(chi2),lam*AA]
            logmetric.append(metric)
            #            print(metric,np.sum(A),np.sum(X))
            import terminalplot
            Xn=cp.asnumpy(X)
            bandl=np.array(range(0,len(Xn[0,:])))
            terminalplot.plot(list(bandl),list(Xn[np.mod(jj,3),:]))

            if np.mod(i,10000)==0:
                LogNMF(i,cp.asnumpy(A),cp.asnumpy(X))
        #----------------------------------------------------
            
        Wt = cp.dot(cp.dot(W.T,Y),X.T)+ epsilon
        Wb = (cp.dot(cp.dot(cp.dot(cp.dot(W.T,W),A),X),X.T)) + lam*2*A + epsilon
        A = A*(Wt/Wb)
        #A = cp.dot(cp.diag(1/cp.sum(A[:,:],axis=1)),A)
        A = cp.dot(cp.diag(1/cp.sum(A[:,:],axis=1)),A)
        
        Wt = cp.dot(cp.dot(A.T,W.T),Y)+ epsilon
        Wb = cp.dot(cp.dot(cp.dot(cp.dot(A.T,W.T),W),A),X)+ 2000.0*X + epsilon 
        X = X*(Wt/Wb)
        #X = cp.dot(cp.diag(1/cp.sum(X[:,:],axis=1)),X)
        #X = cp.dot(X,cp.diag(1/cp.sum(X[:,:],axis=0)))
        
    A=cp.asnumpy(A)
    X=cp.asnumpy(X)
    #----------------------------------------------------
    LogMetricPlot(logmetric)
    #----------------------------------------------------

    return A, X


def NG_MVC_NMF(Ntry,lcall,Win,A0,X0,lam,epsilon):
    Y=cp.asarray(lcall)
    W=cp.asarray(Win)
    A=cp.asarray(A0)
    X=cp.asarray(X0)
    logmetric=[]
    jj=0
    for i in range(0,Ntry):
        ATA = cp.dot(A.T,A)
        #----------------------------------------------------
        if np.mod(i,1000)==0:
            jj=jj+1
            chi2=cp.sum((Y - cp.dot(cp.dot(W,A),X))**2)
            metric=[i,cp.asnumpy(chi2+lam*cp.linalg.det(ATA)),cp.asnumpy(chi2),cp.asnumpy(lam*cp.linalg.det(ATA))]
            logmetric.append(metric)
            #            print(metric,np.sum(A),np.sum(X))
            import terminalplot
            Xn=cp.asnumpy(X)
            bandl=np.array(range(0,len(Xn[0,:])))
            terminalplot.plot(list(bandl),list(Xn[np.mod(jj,3),:]))

            if np.mod(i,10000)==0:
                LogNMF(i,cp.asnumpy(A),cp.asnumpy(X))
        #----------------------------------------------------
            
        Wt = cp.dot(cp.dot(cp.dot(W.T,Y),X.T),ATA)+ epsilon
        Wb = cp.dot(cp.dot(cp.dot(cp.dot(cp.dot(W.T,W),A),X),X.T),ATA) + lam*cp.linalg.det(ATA)*A + epsilon
        A = A*(Wt/Wb)
        #A = cp.dot(cp.diag(1/cp.sum(A[:,:],axis=1)),A)
        X = cp.dot(X,cp.diag(1/cp.sum(X[:,:],axis=0)))
        
        Wt = cp.dot(cp.dot(A.T,W.T),Y)+ epsilon
        Wb = cp.dot(cp.dot(cp.dot(cp.dot(A.T,W.T),W),A),X)+ epsilon 
        X = X*(Wt/Wb)
        #       X = cp.dot(cp.diag(1/cp.sum(X[:,:],axis=1)),X)
        #X = cp.dot(X,cp.diag(1/cp.sum(X[:,:],axis=0)))
        
    A=cp.asnumpy(A)
    X=cp.asnumpy(X)
    #----------------------------------------------------
    LogMetricPlot(logmetric)
    #----------------------------------------------------

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
        ATA = cp.dot(A.T,A)
        chi2=cp.sum((Y - cp.dot(cp.dot(W,A),X))**2)
        if np.mod(i,10)==0: print(i,chi2+lam*cp.linalg.det(ATA),chi2,lam*cp.linalg.det(ATA))
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


def LogNMF(i,A,X):
    import healpy as hp
    import matplotlib.pyplot as plt
    hp.mollview(A[:,0], title="0",flip="geo",cmap=plt.cm.jet)
    plt.savefig("run/mmap0_"+str(i)+".png")
    plt.close()

    hp.mollview(A[:,1], title="1",flip="geo",cmap=plt.cm.jet)
    plt.savefig("run/mmap1_"+str(i)+".png")
    plt.close()

    hp.mollview(A[:,2], title="2",flip="geo",cmap=plt.cm.jet)
    plt.savefig("run/mmap2_"+str(i)+".png")
    plt.close()

    fig= plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    plt.plot(X[0,:],"o",label="Component 0",color="C0")
    plt.plot(X[1,:],"s",label="Component 1",color="C1")
    plt.plot(X[2,:],"^",label="Component 2",color="C2")
    plt.plot(X[0,:],color="C0")
    plt.plot(X[1,:],color="C1")
    plt.plot(X[2,:],color="C2")
    plt.savefig("run/unmix_"+str(i)+".png")
    plt.close()

def LogMetricPlot(logmetric):
    import matplotlib.pyplot as plt

    logmetric=np.array(logmetric)
    fig= plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    plt.plot(logmetric[1:,0],logmetric[1:,1],label="Q")
    plt.plot(logmetric[1:,0],logmetric[1:,2],label="chi")
    plt.plot(logmetric[1:,0],logmetric[1:,3],label="lam det A")
    plt.yscale("log")
    plt.legend()
    plt.savefig("run/logmetric.png")
    plt.close()
