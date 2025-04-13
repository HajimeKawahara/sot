import healpy as hp
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import sys
import scipy

def init_random(N,npix,Y):
    A0=np.random.rand(npix,N)
    X0=np.random.rand(N,np.shape(Y)[1])
    return A0,X0

def init_random_direct(N,Y):
    A0=np.random.rand(np.shape(Y)[0],N)
    X0=np.random.rand(N,np.shape(Y)[1])
    return A0,X0


def initpca(N,W,lcall,lam,mode="Ridge"):
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    pca = PCA(n_components=N)
    pca.fit(lcall) #x(k,l)
    #    X0=(np.abs(pca.components_))
    X0=((pca.components_))
    flo=np.min(X0)
    X0=X0-flo
    iN=np.shape(W)[0]
    jN=np.shape(W)[1]
    kN=np.shape(X0)[0]
    lN=np.shape(X0)[1]
    
    dd=lcall.flatten()
    Wd=np.einsum("ij,kl->iljk",W,X0)
    M=jN*kN
    N=iN*lN
    Wd=Wd.reshape((N,M))

    if mode=="Ridge":
        clf = Ridge(alpha=lam)
        clf.fit(Wd, dd)
        A0=(clf.coef_.reshape((jN,kN)))
        flo=np.min(A0)
        A0=A0-flo
    elif mode=="NNLS-ridge":
        dd=np.concatenate([dd,np.zeros(M)])
        Wd=np.concatenate([Wd,lam*np.eye(M)])
        print("Solve NNLS")
        sol,rnorm=scipy.optimize.nnls(Wd, dd)
        A0=(sol.reshape((jN,kN)))
    elif mode=="Random":
        A0=np.random.rand(jN,kN)
    else:
        print("No mode for initpca.")
        sys.exit()
    return A0,X0

def plotinit(A0):
    #==================================================
    hp.mollview(A0[:,0], title="",flip="geo",cmap=plt.cm.jet)
    plt.savefig("m0.png")
    hp.mollview(A0[:,1], title="",flip="geo",cmap=plt.cm.jet)
    plt.savefig("m1.png")
    hp.mollview(A0[:,2], title="",flip="geo",cmap=plt.cm.jet)
    plt.savefig("m2.png")
