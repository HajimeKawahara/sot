import numpy as np
import sys


def check_nonnegative(Y,lab):
    if np.min(Y)<0:
        print("Error: Negative elements in the initial matrix of "+lab)
        sys.exit()

def QP_NMR(reg,Ntry,lcall,W,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=10,NtryAPGA=1000,eta=0.0,delta=1.e-6,Lipx="norm2",Lipa="frobenius"):
    import scipy
    check_nonnegative(lcall,"LC")
    check_nonnegative(A0,"A")
    check_nonnegative(X0,"X")
    Ni=np.shape(lcall)[0]
    Nl=np.shape(lcall)[1]
    Nk=np.shape(A0)[1]
    Nj=np.shape(A0)[0]

    if reg=="L2-VRDet":
        res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.linalg.det(np.dot(X0,X0.T))
    elif reg=="L2-VRLD":
        nu=1.0
        res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.log(np.linalg.det(np.dot(X0,X0.T)+delta*np.eye(Nk)))        
    elif reg=="Dual-L2":
        res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.sum(X0**2)
    elif reg=="L2":
        res=np.sum((lcall-W@A0@X0)**2)+lamA*np.sum(A0**2)
    else:
        print("No mode. Halt.")
        sys.exit()

    print("Ini residual=",res)
    A=np.copy(A0)
    X=np.copy(X0)
    Y=np.copy(lcall)

    WTW=np.dot(W.T,W)

    jj=0
    resall=[]
    for i in range(0,Ntry):
        print(i)
        ## xk
        for k in range(0,Nk):
            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
            Delta=Y-np.dot(W,AX)
            ak=A[:,k]
            Wa=np.dot(W,ak)
            W_x=np.dot(Wa,Wa)*np.eye(Nl)
            bx=np.dot(np.dot(Delta.T,W),ak)
            if reg=="L2-VRDet":
                Xminus = np.delete(X,obj=k,axis=0)
                XXTinverse=np.linalg.inv(np.dot(Xminus,Xminus.T))
                K=np.eye(Nl) - np.dot(np.dot(Xminus.T,XXTinverse),Xminus)
                K=K*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX
                X[k,:]=APGr(Nl,W_x + K ,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="L2-VRLD":
                E_x=lamX*nu*np.eye(Nl)
                X[k,:]=APGr(Nl,W_x + E_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="Dual-L2":
                T_x=lamX*np.eye(Nj)
                X[k,:]=APGr(Nl,W_x + T_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="L2":
                X[k,:]=APGr(Nl,W_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)

        ## ak
#        for k in range(0,Nk):
#            AX=np.dot(np.delete(A,obj=k,axis=1),np.delete(X,obj=k,axis=0))
#            Delta=Y-np.dot(W,AX)
            xk=X[k,:]
            W_a=(np.dot(xk,xk))*(np.dot(W.T,W))
            b=np.dot(np.dot(W.T,Delta),xk)
            T_a=lamA*np.eye(Nj)
            A[:,k]=APGr(Nj,W_a+T_a,b,A[:,k],Ntry=NtryAPGA, eta=eta, Lip=Lipa)

        Like=(np.sum((Y-np.dot(np.dot(W,A),X))**2))
        RA=(lamA*np.sum(A0**2))
        if reg=="L2-VRDet":
            RX=(lamX*np.linalg.det(np.dot(X,X.T)))
        elif reg=="L2-VRLD":
            eig=np.linalg.eigvals((np.dot(X,X.T) + delta*np.eye(Nk)))
            nu=1.0/np.min(np.abs(eig))
            print("nu=",nu)
            RX=(lamX*np.log(np.linalg.det(np.dot(X,X.T)+delta*np.eye(Nk))))
        elif reg=="Dual-L2":
            RX=(lamX*np.sum(X**2))
        elif reg=="L2":
            RX=0.0
            
        res=Like+RA+RX
        resall.append([res,Like,RA,RX])                
        print("Residual=",res)
        #normalization
        #LogNMF(i,A,X,Nk)
        if np.mod(jj,100) == 0:
            bandl=np.array(range(0,len(X[0,:])))
            import terminalplot
            terminalplot.plot(list(bandl),list(X[np.mod(jj,Nk),:]))

        jj=jj+1
        if np.mod(jj,1000) == 0:
            np.savez(filename+"j"+str(jj),A,X,resall)
            
    return A, X, resall


def APGr(n,Q,p,x0,Ntry=1000,alpha0=0.9,eta=0.0):
    #Accelerated Projected Gradient + restart
    #n=np.shape(Q)[0]
    normQ = np.linalg.norm(Q,2)
    Theta1 = np.eye(n) - Q/normQ
    theta2 = p/normQ
    x = np.copy(x0)
    y = np.copy(x0)
    x[x<0]=0.0
    alpha=alpha0
    cost0=0.5*np.dot(x0,np.dot(Q,x0)) - np.dot(p,x0)
    costp=cost0
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
        elif costp - cost < eta:
            print(i,cost0 - cost)
            return x

        costp=np.copy(cost)
        if cost != cost:
            print("Halt at APGr")
            print("Q,p,x0",Q,p,x0)
            print("cost=",cost)
            sys.exit()
            
    print(i,cost0 - cost)

    return x
