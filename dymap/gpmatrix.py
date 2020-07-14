import numpy as np


def make_tilPi(Nj,K):
    nrow,ncol=np.shape(K)
    C = np.zeros((nrow*Nj,ncol*Nj))
    for j in range(0,Nj):
        C[j*nrow:(j+1)*nrow,j*ncol:(j+1)*ncol]=K
                 
    return C

def make_tilPi_kron(KS,KT):    
    ### SLOW ###
    C = np.kron(KS,KT)
    return C


def make_tilW(Wx):
    Ni,Nj=np.shape(Wx)
    B=np.zeros((Ni,Ni*Nj))
    for j in range(0,Nj):
        B[:,Ni*j:Ni*(j+1)]=np.diag(Wx[:,j])
    return B

