def L2VR_NMF_INITSGD(Ntry,lcall,Win,A0,X0,lamA,lamX,epsilon):
    #intialization by SGD
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
    NGswitch = False
    for i in range(0,Ntry):
        XXT = cp.dot(X,X.T)
        
        #----------------------------------------------------
        if np.mod(i,1000)==0:
            jj=jj+1
            AA=np.sum(A*A)
            detXXT=cp.asnumpy(cp.linalg.det(XXT))

            chi2=cp.sum((Y - cp.dot(cp.dot(W,A),X))**2)
            metric=[i,cp.asnumpy(chi2+lamA*AA+lamX*detXXT),cp.asnumpy(chi2),lamA*AA,lamX*detXXT]
            logmetric.append(metric)
            #            print(metric,np.sum(A),np.sum(X))
            import terminalplot
            Xn=cp.asnumpy(X)
            bandl=np.array(range(0,len(Xn[0,:])))
            print(metric)
            terminalplot.plot(list(bandl),list(Xn[np.mod(jj,Nk),:]))
            if np.mod(i,10000)==0:
                LogNMF(i,cp.asnumpy(A),cp.asnumpy(X),Nk)
            if not NGswitch and chi2 < lamX*detXXT:
                NGswitch = True
        #----------------------------------------------------
            
        Wt = cp.dot(cp.dot(W.T,Y),X.T)+ epsilon
        Wb = (cp.dot(cp.dot(cp.dot(cp.dot(W.T,W),A),X),X.T)) + lamA*A + epsilon
        A = A*(Wt/Wb)
        #A = cp.dot(cp.diag(1/cp.sum(A[:,:],axis=1)),A)
        A = cp.dot(cp.diag(1/cp.sum(A[:,:],axis=1)),A)
        if NGswitch:
            #Natural gradient descent
            XTX = cp.dot(X.T,X)
            detXXT=cp.linalg.det(XXT)
            Wt = cp.dot(cp.dot(cp.dot(A.T,W.T),Y),XTX)+ epsilon
            Wb = cp.dot(cp.dot(cp.dot(cp.dot(cp.dot(A.T,W.T),W),A),X),XTX)+ lamX*detXXT*X + epsilon
        else:
            #Steepest gradient descent
            Wt = (cp.dot(cp.dot(A.T,W.T),Y))+ epsilon
            Wb = (cp.dot(cp.dot(cp.dot(cp.dot(A.T,W.T),W),A),X)) + epsilon 

        X = X*(Wt/Wb)
        #X = cp.dot(cp.diag(1/cp.sum(X[:,:],axis=1)),X)
        #X = cp.dot(X,cp.diag(1/cp.sum(X[:,:],axis=0)))
        
    A=cp.asnumpy(A)
    X=cp.asnumpy(X)
    #----------------------------------------------------
    LogMetricPlot(logmetric)
    #----------------------------------------------------

    return A, X, logmetric

