import healpy as hp
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

def init_random(N,npix,Y):
    A0=np.random.rand(npix,N)
    X0=np.random.rand(N,np.shape(Y)[1])
    return A0,X0

def initpca(N,W,lcall):
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    pca = PCA(n_components=N)
    pca.fit(lcall) #x(k,l)
    X0=(np.abs(pca.components_))
    
    iN=np.shape(W)[0]
    jN=np.shape(W)[1]
    kN=np.shape(X0)[0]
    lN=np.shape(X0)[1]
    
    dd=lcall.flatten()
    Wd=np.einsum("ij,kl->iljk",W,X0)
    Wd=Wd.reshape((iN*lN,jN*kN))
    clf = Ridge(alpha=1.e-3)
    clf.fit(Wd, dd)
    A0=np.abs(clf.coef_.reshape((jN,kN)))
    
    return A0,X0

def plotinit(A0):
    #==================================================
    hp.mollview(A0[:,0], title="",flip="geo",cmap=plt.cm.jet)
    plt.savefig("m0.png")
    hp.mollview(A0[:,1], title="",flip="geo",cmap=plt.cm.jet)
    plt.savefig("m1.png")
    hp.mollview(A0[:,2], title="",flip="geo",cmap=plt.cm.jet)
    plt.savefig("m2.png")
