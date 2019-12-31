import numpy as np
import sys

def check_nonnegative(Y,lab):
    if np.min(Y)<0:
        print("Error: Negative elements in the initial matrix of "+lab)
        sys.exit()


def QP_UNC_NMR(Ntry,lcall,W,A0,X0,lamA,epsilon):
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
    NtryAPG=10

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
            X[k,:]=APGr(W_x,bx,X[k,:],Ntry=NtryAPG)
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
            A[:,k]=APGr(W_a+T_a,b,A[:,k],Ntry=NtryAPG)
            
#        A = np.dot(np.diag(1/np.sum(A[:,:],axis=1)),A)
        res=np.sum((lcall-W@A@X)**2)+lamA*np.sum(A**2)
        print("Residual=",res)

        #normalization
        LogNMF(i,A,X,Nk)
        bandl=np.array(range(0,len(X[0,:])))
        import terminalplot
        terminalplot.plot(list(bandl),list(X[np.mod(jj,Nk),:]))

        jj=jj+1
            
    logmetric=[]
    return A, X, logmetric

def APGr(Q,p,x0,Ntry=1000,alpha0=0.9):
    n=np.shape(Q)[0]
    normQ = np.sqrt(np.sum(Q**2))
    Theta1 = np.eye(n) - Q/normQ
    theta2 = p/normQ
    x = np.copy(x0)
    y = np.copy(x0)
    x[x<0]=0.0
    alpha=alpha0
    costp=0.5*np.dot(x0,np.dot(Q,x0)) - np.dot(p,x0)
    for i in range(0,Ntry):
        xp=np.copy(x)
        x = np.dot(Theta1,y) + theta2
        x[x<0] = 0.0

        dx=x-xp
        aa=alpha*alpha
        beta=alpha*(1.0-alpha)
        alpha=0.5*(np.sqrt(aa*aa + 4*aa) - aa)
        beta=beta/(alpha + aa)
        y=x+beta*dx
        cost=0.5*np.dot(x,np.dot(Q,x)) - np.dot(p,x)
        if cost > costp:
            x = np.dot(Theta1,xp) + theta2
            y = np.copy(x)
            alpha=alpha0
        costp=np.copy(cost)
        if cost != cost:
            print(Q,p,x0)
            print(cost)
            sys.exit()

    return x
