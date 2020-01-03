import cupy as cp
import numpy as np
import sys

def check_nonnegative(Y,lab):
    if np.min(Y)<0:
        print("Error: Negative elements in the initial matrix of "+lab)
        sys.exit()

def QP_DET_NMR(Ntry,lcall,W,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=10,NtryAPGA=1000,eta=0.0):
    import scipy
    check_nonnegative(lcall,"LC")
    check_nonnegative(A0,"A")
    check_nonnegative(X0,"X")
    res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.linalg.det(np.dot(X0,X0.T))
    print("Ini residual=",res)
    Ni=np.shape(lcall)[0]
    Nl=np.shape(lcall)[1]
    Nk=np.shape(A0)[1]
    Nj=np.shape(A0)[0]
    Y=cp.asarray(lcall)
    W=cp.asarray(Win)
    A=cp.asarray(A0)
    X=cp.asarray(X0)

    WTW=cp.dot(W.T,W)

    jj=0
    for i in range(0,Ntry):
        print(i)
        ## xk
        for k in range(0,Nk):
#            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
            AX=cp.dot(A,X) - cp.dot(A[:,k:k+1],X[k:k+1,:]) 
            Delta=Y-cp.dot(W,AX)
            ak=A[:,k]
            Wa=cp.dot(W,ak)
            W_x=cp.dot(Wa,Wa)*cp.eye(Nl)
            bx=cp.dot(cp.dot(Delta.T,W),ak)
            #Det XXT
            Xminus = np.delete(cp.asnumpy(X),obj=k,axis=0)
            XXTinverse=cp.linalg.inv(cp.dot(X,X.T)-cp.dot(X[k:k+1,:],X[k:k+1,:].T))
            Kn=cp.eye(Nl) - cp.dot(np.dot(Xminus.T,XXTinverse),Xminus)
            Kn=Kn*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX
            K=cp.asarray(Kn)
            X[k,:]=APGr(Nl,W_x + K ,bx,X[k,:],Ntry=NtryAPGX, eta=eta)

        ## ak
        for k in range(0,Nk):
#            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
            AX=cp.dot(A,X) - cp.dot(A[:,k:k+1],X[k:k+1,:]) 

            Delta=Y-cp.dot(W,AX)
            xk=X[k,:]
            W_a=(cp.dot(xk,xk))*(cp.dot(W.T,W))
            b=cp.dot(cp.dot(W.T,Delta),xk)
            T_a=lamA*cp.eye(Nj)
            A[:,k]=APGr(Nj,W_a+T_a,b,A[:,k],Ntry=NtryAPGA, eta=eta)
            
        #res=np.sum((lcall-W@A@X)**2)+lamA*np.sum(A**2)+lamX*np.linalg.det(np.dot(X,X.T))
        print("Residual=",res)
        #normalization
        #LogNMF(i,A,X,Nk)
        bandl=np.array(range(0,len(X[0,:])))
        import terminalplot
        terminalplot.plot(list(bandl),list(cp.asnumpy(X[np.mod(jj,Nk),:])))

        jj=jj+1
        if np.mod(jj,1000) == 0:
            np.savez(filename+"j"+str(jj),cp.asunmpy(A),cp.asnumpy(X))

            
    logmetric=[]
    return A, X, logmetric

def QP_L2_NMR(Ntry,lcall,W,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=10,NtryAPGA=1000,eta=0.0):
    import scipy
    check_nonnegative(lcall,"LC")
    check_nonnegative(A0,"A")
    check_nonnegative(X0,"X")
    res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.sum(X0**2)
    print("Ini residual=",res)
    A=np.copy(A0)
    X=np.copy(X0)
    Y=np.copy(lcall)
    Ni=np.shape(Y)[0]
    Nl=np.shape(Y)[1]
    Nk=np.shape(A)[1]
    Nj=np.shape(A)[0]

    WTW=np.dot(W.T,W)

    jj=0
    for i in range(0,Ntry):
        print(i)
        ## xk
        for k in range(0,Nk):
            print("X, k=",k)
            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
            Delta=Y-np.dot(W,AX)
            ak=A[:,k]
            Wa=np.dot(W,ak)
            W_x=np.dot(Wa,Wa)*np.eye(Nl)
            bx=np.dot(np.dot(Delta.T,W),ak)
            T_x=lamX*np.eye(Nl)

            Xpropose=APGr(W_x+T_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta)
            if np.sum(Xpropose) > 0.0:
                X[k,:]=Xpropose
            else:
                print("Zero end")
                sys.exit()
        ## ak
        res=np.sum((lcall-W@A@X)**2)+lamA*np.sum(A**2)
        for k in range(0,Nk):
            print("A, k=",k)
            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
            Delta=Y-np.dot(W,AX)
            xk=X[k,:]
            W_a=(np.dot(xk,xk))*(np.dot(W.T,W))
            b=np.dot(np.dot(W.T,Delta),xk)
            T_a=lamA*np.eye(Nj)
            Apropose=APGr(W_a+T_a,b,A[:,k],Ntry=NtryAPGA, eta=eta)
            if np.sum(Xpropose) > 0.0:
                A[:,k]=Apropose
            else:
                print("Zero end")
                sys.exit()

        #####################
#        nprev=(np.sum(W@A@X))
#        X = np.dot(np.diag(1/np.sum(X[:,:],axis=1)),X)
#        A = np.dot(np.diag(1/np.sum(A[:,:],axis=1)),A)
#        nup=(np.sum(W@A@X))
#        X=X*nprev/nup
        #####################
        res=np.sum((lcall-W@A@X)**2)+lamA*np.sum(A**2)+lamX*np.sum(X**2)
        print("Residual=",res)

        #normalization
        LogNMF(i,A,X,Nk)
        bandl=np.array(range(0,len(X[0,:])))
        import terminalplot
        terminalplot.plot(list(bandl),list(X[np.mod(jj,Nk),:]))

        jj=jj+1
        if np.mod(jj,1000) == 0:
            np.savez(filename+"j"+str(jj),A,X)
            
    logmetric=[]
    return A, X, logmetric

def QP_UNC_NMR(Ntry,lcall,W,A0,X0,lamA,epsilon,filename,NtryAPGX=10,NtryAPGA=1000,eta=0.0):
    import scipy
    check_nonnegative(lcall,"LC")
    check_nonnegative(A0,"A")
    check_nonnegative(X0,"X")
    res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)
    print("Ini residual=",res)
    A=np.copy(A0)
    X=np.copy(X0)
    Y=np.copy(lcall)
    Ni=np.shape(Y)[0]
    Nl=np.shape(Y)[1]
    Nk=np.shape(A)[1]
    Nj=np.shape(A)[0]

    WTW=np.dot(W.T,W)

    jj=0
    for i in range(0,Ntry):
        print(i)
        ## xk
        for k in range(0,Nk):
            print("X, k=",k)
            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
            Delta=Y-np.dot(W,AX)
            ak=A[:,k]
            Wa=np.dot(W,ak)
            W_x=np.dot(Wa,Wa)*np.eye(Nl)
            bx=np.dot(np.dot(Delta.T,W),ak)
            Xpropose=APGr(W_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta)
            if np.sum(Xpropose) > 0.0:
                X[k,:]=Xpropose
            else:
                print("Zero end")
                sys.exit()
        ## ak
        res=np.sum((lcall-W@A@X)**2)+lamA*np.sum(A**2)
        for k in range(0,Nk):
            print("A, k=",k)
            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
            Delta=Y-np.dot(W,AX)
            xk=X[k,:]
            W_a=(np.dot(xk,xk))*(np.dot(W.T,W))
            b=np.dot(np.dot(W.T,Delta),xk)
            T_a=lamA*np.eye(Nj)
            Apropose=APGr(W_a+T_a,b,A[:,k],Ntry=NtryAPGA, eta=eta)
            if np.sum(Xpropose) > 0.0:
                A[:,k]=Apropose
            else:
                print("Zero end")
                sys.exit()

        #####################
#        nprev=(np.sum(W@A@X))
#        X = np.dot(np.diag(1/np.sum(X[:,:],axis=1)),X)
#        A = np.dot(np.diag(1/np.sum(A[:,:],axis=1)),A)
#        nup=(np.sum(W@A@X))
#        X=X*nprev/nup
        #####################
        res=np.sum((lcall-W@A@X)**2)+lamA*np.sum(A**2)
        print("Residual=",res)

        #normalization
        LogNMF(i,A,X,Nk)
        bandl=np.array(range(0,len(X[0,:])))
        import terminalplot
        terminalplot.plot(list(bandl),list(X[np.mod(jj,Nk),:]))

        jj=jj+1
        if np.mod(jj,1000) == 0:
            np.savez(filename+"j"+str(jj),A,X)
            
    logmetric=[]
    return A, X, logmetric

def APGr(n,Q,p,x0,Ntry=1000,alpha0=0.9,eta=0.0):
    #Accelerated Projected Gradient + restart
#    n=np.shape(Q)[0]
    normQ = cp.sqrt(cp.sum(Q**2))
    Theta1 = cp.eye(n) - Q/normQ
    theta2 = p/normQ
    x = cp.copy(x0)
    y = cp.copy(x0)
    x[x<0]=0.0
    alpha=alpha0
    cost0=0.5*cp.dot(x0,cp.dot(Q,x0)) - cp.dot(p,x0)
    costp=cost0
    for i in range(0,Ntry):
        xp=cp.copy(x)
        x = cp.dot(Theta1,y) + theta2
        x[x<0] = 0.0
        dx=x-xp
        aa=alpha*alpha
        beta=alpha*(1.0-alpha)
        alpha=0.5*(np.sqrt(aa*aa + 4*aa) - aa)
        beta=beta/(alpha + aa)
        y=x+beta*dx
        cost=0.5*cp.dot(x,cp.dot(Q,x)) - cp.dot(p,x)
        if cost > costp:
            x = cp.dot(Theta1,xp) + theta2
            y = cp.copy(x)
            alpha=alpha0
        elif costp - cost < eta:
            print(i,cost0 - cost)
            return x

        costp=cp.copy(cost)
        if cost != cost:
            print("Halt at APGr")
            print("Q,p,x0",Q,p,x0)
            print("cost=",cost)
            sys.exit()
            
    print(i,cost0 - cost)

    return x
        
def MP_L2VR_NMF(Ntry,lcall,Win,A0,X0,lamA,lamX,epsilon,rho=0.1, off=0):
    #multiplicative L2VR using natural gradient
    check_nonnegative(lcall,"LC")
    check_nonnegative(A0,"A")
    check_nonnegative(X0,"X")
    Nk=np.shape(A0)[1]
    Y=cp.asarray(lcall)
    W=cp.asarray(Win)
    A=cp.asarray(A0)
    X=cp.asarray(X0)
    logmetric=[]
    jj=0
    for i in range(0,Ntry):
        ATA = cp.dot(A.T,A)
        XTX = cp.dot(X.T,X)
        XXT = cp.dot(X,X.T)

        #----------------------------------------------------
        if np.mod(i,1000)==0:
            jj=jj+1
            AA=np.sum(A*A)
            detXXT=cp.asnumpy(cp.linalg.det(XXT))
            chi2=cp.sum((Y - cp.dot(cp.dot(W,A),X))**2)
            metric=[i+off,cp.asnumpy(chi2+lamA*AA+lamX*detXXT),cp.asnumpy(chi2),lamA*AA,lamX*detXXT]
            logmetric.append(metric)
            #            print(metric,np.sum(A),np.sum(X))
            import terminalplot
            Xn=cp.asnumpy(X)
            bandl=np.array(range(0,len(Xn[0,:])))
            print(metric)
            terminalplot.plot(list(bandl),list(Xn[np.mod(jj,Nk),:]))
            if np.mod(i,10000)==0:
                LogNMF(i+off,cp.asnumpy(A),cp.asnumpy(X),Nk)
        #----------------------------------------------------

        detXXT=cp.linalg.det(XXT)
        WA=cp.dot(W,A)
        Wt = cp.dot(cp.dot(WA.T,Y),XTX) + epsilon
        Wb = cp.dot(cp.dot(cp.dot(WA.T,WA),X),XTX)+ lamX*detXXT*X + epsilon
        #SGD 
        Wt = cp.dot(cp.dot(W.T,Y),X.T)+ epsilon
        Wb = (cp.dot(cp.dot(cp.dot(W.T,WA),X),X.T)) + lamA*A + epsilon
        A = A*(Wt/Wb)
        A = cp.dot(cp.diag(1/cp.sum(A[:,:],axis=1)),A)
        
    A=cp.asnumpy(A)
    X=cp.asnumpy(X)
    #----------------------------------------------------
    LogMetricPlot(logmetric)
    #----------------------------------------------------

    return A, X, logmetric

        
def MP_L2_NMF(Ntry,lcall,Win,A0,X0,lamA,lamX,epsilon,off=0):
    #multiplicative L2
    check_nonnegative(lcall,"LC")
    check_nonnegative(A0,"A")
    check_nonnegative(X0,"X")
    Nk=np.shape(A0)[1]
    Y=cp.asarray(lcall)
    W=cp.asarray(Win)
    A=cp.asarray(A0)
    X=cp.asarray(X0)
    logmetric=[]
    jj=0
    for i in range(0,Ntry):
        
        #----------------------------------------------------
        if np.mod(i,1000)==0:
            jj=jj+1
            AA=np.sum(A*A)
            XX=np.sum(X*X)
            chi2=cp.sum((Y - cp.dot(cp.dot(W,A),X))**2)
            metric=[i+off,cp.asnumpy(chi2+lamA*AA+lamX*XX),cp.asnumpy(chi2),lamA*AA,lamX*XX]
            logmetric.append(metric)
            #            print(metric,np.sum(A),np.sum(X))
            import terminalplot
            Xn=cp.asnumpy(X)
            bandl=np.array(range(0,len(Xn[0,:])))
            print(metric)
            terminalplot.plot(list(bandl),list(Xn[np.mod(jj,Nk),:]))
            if np.mod(i,10000)==0:
                LogNMF(i+off,cp.asnumpy(A),cp.asnumpy(X),Nk)
        #----------------------------------------------------
            
        Wt = cp.dot(cp.dot(W.T,Y),X.T)+ epsilon
        Wb = (cp.dot(cp.dot(cp.dot(cp.dot(W.T,W),A),X),X.T)) + lamA*A + epsilon
        A = A*(Wt/Wb)
        A = cp.dot(cp.diag(1/cp.sum(A[:,:],axis=1)),A)

        WA=cp.dot(W,A)        
        Wt = cp.dot(WA.T,Y)+ epsilon
        Wb = cp.dot(cp.dot(WA.T,WA),X)+ lamX*X + epsilon 
        X = X*(Wt/Wb)
        
    A=cp.asnumpy(A)
    X=cp.asnumpy(X)
    #----------------------------------------------------
    LogMetricPlot(logmetric)
    #----------------------------------------------------

    return A, X, logmetric



def LogNMF(i,A,X, lim=3):
    import healpy as hp
    import matplotlib.pyplot as plt

    for j in range(0,lim):
        hp.mollview(A[:,j], title="Component "+str(j),flip="geo",cmap=plt.cm.jet)
        plt.savefig("run/mmap"+str(j)+"_"+str(i)+".png")
        plt.close()

    fig= plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    marker=["o","s","^"]
    for j in range(0,lim):
        plt.plot(X[j,:],marker[np.mod(j,3)],label="Component "+str(j),color="C"+str(j))
        plt.plot(X[j,:],color="C"+str(j))
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
