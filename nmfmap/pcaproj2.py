import io_surface_type 
import io_refdata
import toymap
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import matplotlib
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
import plotmap as pm
import read_data

def tik(lcall,W):
    ### simple Tikhonov
    U,S,VT=np.linalg.svd(W)
    unwrap=[]
    #clfx = lm.Ridge(alpha=1.e-2)
    lambdatik=1.e-2
    p=min(np.shape(lcall)[0],np.shape(W)[1])
    for j in range(0,np.shape(lcall)[1]):
        Sm=np.zeros(np.shape(W)[1])
        #    clfx.fit(W, lcall[:,i])
        #    hest=clfx.coef_
        dv=lcall[:,j]
        for i in range(0,p):
            phij=(S[i]/(S[i]**2+lambdatik**2))
            Sm=Sm+phij*np.inner(U[:,i],dv)*VT[i,:]
        unwrap.append(Sm)
    unwrap=np.array(unwrap).T
    print(np.shape(unwrap))
    return unwrap

def projpc(Xin,Xp):
    pc1=Xp[0,:]
    pc2=Xp[1,:]
    xpc1=np.dot(Xin,pc1)
    xpc2=np.dot(Xin,pc2)
    return xpc1,xpc2

if __name__=='__main__':
    import sys
    #    axfile="npz/T116/T116_L2-VRLD_A-2.0X4.0j99000.npz"
    fontsize=16
    matplotlib.rcParams.update({'font.size':fontsize})
    fig=plt.figure(figsize=(7,5))
    W=np.load("w.npz")["arr_0"]
    lcall=np.load("lcall.npz")["arr_0"]
    pca = PCA(n_components=2)
    pca.fit(lcall) #x(k,l)
    Xp=((pca.components_))
  
    for l in range(1,3):
        axfile=sys.argv[l]
        print(axfile)
        A,X,resall=read_data.readax(axfile)
        WA=np.dot(W,A)
        lcall=np.load("lcall.npz")["arr_0"]
        for i in range(0,np.shape(lcall)[1]):
            lcall[:,i]=lcall[:,i]/np.sum(WA,axis=1)

        xpc1,xpc2=projpc(X,Xp)
        lcpc1,lcpc2=projpc(lcall,Xp)
        
        #unwrap=tik(lcall,W)
        #cpc1,cpc2=projpc(unwrap,Xp)
        
        As=np.sum(A,axis=1)
        for k in range(0,np.shape(WA)[1]):
            A[:,k]=A[:,k]/As
            B=np.dot(A,X)
        bpc1,bpc2=projpc(B,Xp)
        
#        plt.plot(lcpc1,lcpc2,"+", label="Light curve")        
        plt.plot(bpc1,bpc2,".",alpha=0.3, label="Disentangled")
        #plt.plot(cpc1,cpc2,".",alpha=0.3)    
#        for i in range(0,len(xpc1)):
#            plt.text(xpc1[i],xpc2[i],str(i),fontsize=18)
#        plt.plot(xpc1,xpc2,"o",color="red",label="endmembers")
#        plt.plot(np.concatenate([xpc1,xpc1]),np.concatenate([xpc2,xpc2]),color="gray")


    plt.xlabel("PC1")
    plt.ylabel("PC2")
#    plt.legend()        
    plt.savefig("pcaproj.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()
