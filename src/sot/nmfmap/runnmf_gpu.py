import cupy as cp
import numpy as np
import sys
import time
from sot.nmfmap import apg

def check_nonnegative(Y,lab):
    if np.min(Y)<0:
        print("Error: Negative elements in the initial matrix of "+lab)
        sys.exit()

def GPQP_GNMF(reg,Ntry,lcall,Win,A0,X0,KSin,lamX,epsilon,filename,NtryAPGX=10,NtryAPGA=1000,eta=0.0, delta=1.e-6, off=0, nu=1.0,Lipx="norm2",Lipa="frobenius", endc=1.e-5,Nsave=10000,semiNMF=False):
    """
    Summary
    --------------
    GPQP_GNMF: Gaussian Process and Quadratic Programing for Geometric Non-negative Matrix Factorization

    """

    import scipy
    check_nonnegative(lcall,"LC")
    check_nonnegative(A0,"A")
    check_nonnegative(X0,"X")
    Ni=np.shape(lcall)[0]
    Nl=np.shape(lcall)[1]
    Nk=np.shape(A0)[1]
    Nj=np.shape(A0)[0]

    invKS0=np.linalg.inv(KSin)
    if reg=="GP-VRDet":
        res=np.sum((lcall-Win@A0@X0)**2)+np.trace(A0.T@invKS0@A0)+lamX*np.linalg.det(np.dot(X0,X0.T))
    elif reg=="GP-VRLD":
        res=np.sum((lcall-Win@A0@X0)**2)+np.trace(A0.T@invKS0@A0)+lamX*np.log(np.linalg.det(np.dot(X0,X0.T)+delta*np.eye(Nk)))
    elif reg=="GP-L2":
        res=np.sum((lcall-Win@A0@X0)**2)+np.trace(A0.T@invKS0@A0)+lamX*np.sum(X0**2)
    elif reg=="GP":
        res=np.sum((lcall-Win@A0@X0)**2)+np.trace(A0.T@invKS0@A0)
    else:
        print("No mode. Halt.")
        sys.exit()
        
    print("Ini residual=",res)
    Y=cp.asarray(lcall)
    W=cp.asarray(Win)
    A=cp.asarray(A0)
    X=cp.asarray(X0)
    WTW=cp.dot(W.T,W)

    jj=off
    resall=[]
    
    Kw=cp.asarray(Win@KSin@Win.T)
    KS=cp.asarray(KSin)
    invKS=np.linalg.inv(KS)

    for i in range(0,Ntry):
        print(i)

        ## xk
        for k in range(0,Nk):
            AX=cp.dot(A,X) - cp.dot(A[:,k:k+1],X[k:k+1,:])
            Delta=Y-cp.dot(W,AX)
            ak=A[:,k]
            Wa=cp.dot(W,ak)
            W_x=cp.dot(Wa,Wa)*cp.eye(Nl)
            bx=cp.dot(cp.dot(Delta.T,W),ak)
            if reg=="GP-VRDet":
                Xn=cp.asnumpy(X)
                Xminus = np.delete(Xn,obj=k,axis=0)
                XXTinverse=np.linalg.inv(np.dot(Xminus,Xminus.T))
                Kn=np.eye(Nl) - np.dot(np.dot(Xminus.T,XXTinverse),Xminus)
                Kn=Kn*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX
                D_x=cp.asarray(Kn)
                X[k,:]=apg.APGr(Nl,W_x + D_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="GP-VRLD":
                E_x=lamX*nu*cp.eye(Nl)
                X[k,:]=apg.APGr(Nl,W_x + E_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="GP-L2":
                T_x=lamX*cp.eye(Nl)
                X[k,:]=apg.APGr(Nl,W_x + T_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="GP":
                X[k,:]=apg.APGr(Nl,W_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)

            ## X normalization
            #X[k,:]=X[k,:]/cp.sum(X[k,:])
            
            ## ak            
            xk=X[k,:]
            ck=cp.sum(xk**2)
            lk=cp.sum(Delta*xk,axis=1)/ck
            IKw=cp.eye(Ni)+ck*Kw        
            Xlc=cp.linalg.solve(IKw,lk)
            atmp= ck*cp.dot(cp.dot(KS,W.T),Xlc)
#            atmp[atmp<0.0]=0.0
#            A[:,k]=atmp
                        
        Like=cp.asnumpy(cp.sum((Y-cp.dot(cp.dot(W,A),X))**2))
        RA=cp.asnumpy(cp.trace(A.T@invKS@A))
        if reg=="GP-VRDet":
            RX=cp.asnumpy(lamX*cp.linalg.det(cp.dot(X,X.T)))
        elif reg=="GP-VRLD":
            eig=np.linalg.eigvals(cp.asnumpy(cp.dot(X,X.T) + delta*cp.eye(Nk)))
            nu=1.0/np.min(np.abs(eig))
            print("nu=",nu)
            RX=cp.asnumpy(lamX*cp.log(cp.linalg.det(cp.dot(X,X.T)+delta*cp.eye(Nk))))
        elif reg=="GP-L2":
            RX=cp.asnumpy(lamX*cp.sum(X**2))
        elif reg=="GP":
            RX=0.0
        resprev=res

        res=Like+RA+RX
        diff=resprev - res
        resall.append([res,Like,RA,RX])                
        print("Residual=",res,Like,RA,RX)
        print("Xave",cp.mean(X))
        print("Aave",cp.mean(A))

        #LogNMF(i,A,X,Nk)
        if np.mod(jj,10)==0:
            bandl=np.array(range(0,len(X[0,:])))
            import terminalplot
            terminalplot.plot(list(bandl),list(cp.asnumpy(X[np.mod(jj,Nk),:])))

        jj=jj+1
        if np.mod(jj,Nsave) == 0:
            np.savez(filename+"j"+str(jj),cp.asnumpy(A),cp.asnumpy(X),resall)
        if diff < endc:
            np.savez(filename+"Ej"+str(jj),cp.asnumpy(A),cp.asnumpy(X),resall)
            return cp.asnumpy(A),cp.asnumpy(X), resall
        
    return cp.asnumpy(A),cp.asnumpy(X), resall

        
#
# QP_GNMF: Quadratic Programing for Geometric Non-negative Matrix Factorization
#
def QP_GNMF(reg,Ntry,lcall,Win,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=10,NtryAPGA=1000,eta=0.0, delta=1.e-6, off=0, nu=1.0,Lipx="norm2",Lipa="frobenius", endc=1.e-5,Nsave=10000,semiNMF=False):
    import scipy
    check_nonnegative(lcall,"LC")
    check_nonnegative(A0,"A")
    check_nonnegative(X0,"X")
    Ni=np.shape(lcall)[0]
    Nl=np.shape(lcall)[1]
    Nk=np.shape(A0)[1]
    Nj=np.shape(A0)[0]

    if reg=="L2-VRDet":
        res=np.sum((lcall-Win@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.linalg.det(np.dot(X0,X0.T))
    elif reg=="L2-VRLD":
        res=np.sum((lcall-Win@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.log(np.linalg.det(np.dot(X0,X0.T)+delta*np.eye(Nk)))
    elif reg=="Dual-L2":
        res=np.sum((lcall-Win@A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.sum(X0**2)
    elif reg=="L2":
        res=np.sum((lcall-Win@A0@X0)**2)+lamA*np.sum(A0**2)
    else:
        print("No mode. Halt.")
        sys.exit()
        
    print("Ini residual=",res)
    Y=cp.asarray(lcall)
    W=cp.asarray(Win)
    A=cp.asarray(A0)
    X=cp.asarray(X0)
    WTW=cp.dot(W.T,W)

    jj=off
    resall=[]
    for i in range(0,Ntry):
        print(i)

        ## xk
        for k in range(0,Nk):
            AX=cp.dot(A,X) - cp.dot(A[:,k:k+1],X[k:k+1,:])
            Delta=Y-cp.dot(W,AX)
            ak=A[:,k]
            Wa=cp.dot(W,ak)
            W_x=cp.dot(Wa,Wa)*cp.eye(Nl)
            bx=cp.dot(cp.dot(Delta.T,W),ak)
            if reg=="L2-VRDet":
                Xn=cp.asnumpy(X)
                Xminus = np.delete(Xn,obj=k,axis=0)
                XXTinverse=np.linalg.inv(np.dot(Xminus,Xminus.T))
                Kn=np.eye(Nl) - np.dot(np.dot(Xminus.T,XXTinverse),Xminus)
                Kn=Kn*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX
                D_x=cp.asarray(Kn)
                X[k,:]=apg.APGr(Nl,W_x + D_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="L2-VRLD":
                E_x=lamX*nu*cp.eye(Nl)
                X[k,:]=apg.APGr(Nl,W_x + E_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="Dual-L2":
                T_x=lamX*cp.eye(Nl)
                X[k,:]=apg.APGr(Nl,W_x + T_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)
            elif reg=="L2":
                X[k,:]=apg.APGr(Nl,W_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx)

            ## X normalization
            #X[k,:]=X[k,:]/cp.sum(X[k,:])
            
            ## ak
            xk=X[k,:]
            W_a=(cp.dot(xk,xk))*(cp.dot(W.T,W))
            b=cp.dot(cp.dot(W.T,Delta),xk)
            T_a=lamA*cp.eye(Nj)
            if semiNMF:
                A[:,k]=apg.AGr(Nj,W_a+T_a,b,A[:,k],Ntry=NtryAPGA, eta=eta, Lip=Lipa)
            else:
                A[:,k]=apg.APGr(Nj,W_a+T_a,b,A[:,k],Ntry=NtryAPGA, eta=eta, Lip=Lipa)
            ## A normalization
            #A[:,k]=A[:,k]/cp.sum(A[:,k])*Nj
            ## A normalization
            #for k in range(0,Nk):
            #    sumk=cp.sum(A[:,:],axis=1)
            #    sumav=cp.mean(sumk)
            #    A[:,k]=A[:,k]/sumk*sumav

        Like=cp.asnumpy(cp.sum((Y-cp.dot(cp.dot(W,A),X))**2))

        ### BUG FIXED 2020/12/22
        #        RA=cp.asnumpy(lamA*cp.sum(A0**2))
        RA=cp.asnumpy(lamA*cp.sum(A**2))
        ###
        
        if reg=="L2-VRDet":
            RX=cp.asnumpy(lamX*cp.linalg.det(cp.dot(X,X.T)))
        elif reg=="L2-VRLD":
            eig=np.linalg.eigvals(cp.asnumpy(cp.dot(X,X.T) + delta*cp.eye(Nk)))
            nu=1.0/np.min(np.abs(eig))
            print("nu=",nu)
            RX=cp.asnumpy(lamX*cp.log(cp.linalg.det(cp.dot(X,X.T)+delta*cp.eye(Nk))))
        elif reg=="Dual-L2":
            RX=cp.asnumpy(lamX*cp.sum(X**2))
        elif reg=="L2":
            RX=0.0
        resprev=res
        res=Like+RA+RX
        diff=resprev - res
        resall.append([res,Like,RA,RX])                
        print("Residual=",res,Like,RA,RX)
        print("Xave",cp.mean(X))
        print("Aave",cp.mean(A))

        #LogNMF(i,A,X,Nk)
        if np.mod(jj,10)==0:
            bandl=np.array(range(0,len(X[0,:])))
            import terminalplot
            terminalplot.plot(list(bandl),list(cp.asnumpy(X[np.mod(jj,Nk),:])))

        jj=jj+1
        if np.mod(jj,Nsave) == 0:
            np.savez(filename+"j"+str(jj),cp.asnumpy(A),cp.asnumpy(X),resall)
        if diff < endc:
            np.savez(filename+"Ej"+str(jj),cp.asnumpy(A),cp.asnumpy(X),resall)
            return cp.asnumpy(A),cp.asnumpy(X), resall
        
    return cp.asnumpy(A),cp.asnumpy(X), resall

#
# QP_NMF: Quadratic Programing for Non-negative Matrix Factorization
#
def QP_NMF(reg,Ntry,lcall,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=10,NtryAPGA=1000,eta=0.0, delta=1.e-6, off=0, nu=1.0,Lipx="norm2",Lipfacx=1.0,Lipa="frobenius",Lipfaca=1.e-1, endc=1.e-5,Nsave=10000,semiNMF=False):
    import scipy
    check_nonnegative(lcall,"LC")
    check_nonnegative(A0,"A")
    check_nonnegative(X0,"X")
    Ni=np.shape(lcall)[0]
    Nl=np.shape(lcall)[1]
    Nk=np.shape(A0)[1]

    if reg=="L2-VRDet":
        res=np.sum((lcall-A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.linalg.det(np.dot(X0,X0.T))
    elif reg=="L2-VRLD":
        res=np.sum((lcall-A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.log(np.linalg.det(np.dot(X0,X0.T)+delta*np.eye(Nk)))
    elif reg=="Dual-L2":
        res=np.sum((lcall-A0@X0)**2)+lamA*np.sum(A0**2)+lamX*np.sum(X0**2)
    elif reg=="L2":
        res=np.sum((lcall-A0@X0)**2)+lamA*np.sum(A0**2)
    else:
        print("No mode. Halt.")
        sys.exit()
        
    print("Ini residual=",res)
    Y=cp.asarray(lcall)
    A=cp.asarray(A0)
    X=cp.asarray(X0)

    jj=off
    resall=[]
    for i in range(0,Ntry):
        print(i)

        ## xk
        for k in range(0,Nk):
            AX=cp.dot(A,X) - cp.dot(A[:,k:k+1],X[k:k+1,:])
            Delta=Y-AX
            ak=A[:,k]
            W_x=cp.dot(ak,ak)*cp.eye(Nl)
            bx=cp.dot(Delta.T,ak)
            if reg=="L2-VRDet":
                Xn=cp.asnumpy(X)
                Xminus = np.delete(Xn,obj=k,axis=0)
                XXTinverse=np.linalg.inv(np.dot(Xminus,Xminus.T))
                Kn=np.eye(Nl) - np.dot(np.dot(Xminus.T,XXTinverse),Xminus)
                Kn=Kn*np.linalg.det(np.dot(Xminus,Xminus.T))*lamX
                D_x=cp.asarray(Kn)
                X[k,:]=apg.APGr(Nl,W_x + D_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx,Lipfac=Lipfacx)
            elif reg=="L2-VRLD":
                E_x=lamX*nu*cp.eye(Nl)
                X[k,:]=apg.APGr(Nl,W_x + E_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx,Lipfac=Lipfacx)
            elif reg=="Dual-L2":
                T_x=lamX*cp.eye(Nl)
                X[k,:]=apg.APGr(Nl,W_x + T_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx,Lipfac=Lipfacx)
            elif reg=="L2":
                X[k,:]=apg.APGr(Nl,W_x,bx,X[k,:],Ntry=NtryAPGX, eta=eta, Lip=Lipx,Lipfac=Lipfacx)
            
            ## ak
            xk=X[k,:]
            W_a=(cp.dot(xk,xk))
            b=cp.dot(Delta,xk)
            T_a=lamA*cp.eye(Ni)
            if semiNMF:
                A[:,k]=apg.AGr(Ni,W_a+T_a,b,A[:,k],Ntry=NtryAPGA, eta=eta, Lip=Lipa,Lipfac=Lipfaca)
            else:
                A[:,k]=apg.APGr(Ni,W_a+T_a,b,A[:,k],Ntry=NtryAPGA, eta=eta, Lip=Lipa,Lipfac=Lipfaca)
        Like=cp.asnumpy(cp.sum((Y-cp.dot(A,X))**2))
        RA=cp.asnumpy(lamA*cp.sum(A0**2))
        if reg=="L2-VRDet":
            RX=cp.asnumpy(lamX*cp.linalg.det(cp.dot(X,X.T)))
        elif reg=="L2-VRLD":
            eig=np.linalg.eigvals(cp.asnumpy(cp.dot(X,X.T) + delta*cp.eye(Nk)))
            nu=1.0/np.min(np.abs(eig))
            print("nu=",nu)
            RX=cp.asnumpy(lamX*cp.log(cp.linalg.det(cp.dot(X,X.T)+delta*cp.eye(Nk))))
        elif reg=="Dual-L2":
            RX=cp.asnumpy(lamX*cp.sum(X**2))
        elif reg=="L2":
            RX=0.0
        resprev=res
        res=Like+RA+RX
        diff=resprev - res
        resall.append([res,Like,RA,RX])                
        print("Residual=",res,Like,RA,RX)
        print("Xave",cp.mean(X))
        print("Aave",cp.mean(A))
#        print("A=",A)
        #LogNMF(i,A,X,Nk)
        if np.mod(jj,10)==0:
            bandl=np.array(range(0,len(X[0,:])))
            #import terminalplot
            #terminalplot.plot(list(bandl),list(cp.asnumpy(X[np.mod(jj,Nk),:])))

        jj=jj+1
        if np.mod(jj,Nsave) == 0:
            np.savez(filename+"j"+str(jj),cp.asnumpy(A),cp.asnumpy(X),resall)
        if diff < endc:
            print("diff=",diff)
            np.savez(filename+"Ej"+str(jj),cp.asnumpy(A),cp.asnumpy(X),resall)
            return cp.asnumpy(A),cp.asnumpy(X), resall
        
    return cp.asnumpy(A),cp.asnumpy(X), resall



#
# MP_L2VR_GNMF: multiplicative L2VR using natural gradient
#
def MP_L2VR_GNMF(Ntry,lcall,Win,A0,X0,lamA,lamX,epsilon,rho=0.1, off=0):
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

#
# MP_L2_GNMF multiplicative L2
#       
def MP_L2_GNMF(Ntry,lcall,Win,A0,X0,lamA,lamX,epsilon,off=0):
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
