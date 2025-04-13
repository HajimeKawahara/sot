import time
import numpy as np
import io_surface_type 
import io_refdata
import toymap
import mocklc
import matplotlib.pyplot as plt
import healpy as hp
import cupy as cp
import sys
import healpy as hp
import initnmf
#import runnmf_cpu as runnmf #CPU version (slow)
import runnmf_gpu as runnmf #GPU version
from sklearn.decomposition import PCA

np.random.seed(34)

Ain=np.abs(np.random.rand(100,3))
Xin=np.abs(np.random.rand(3,7))
#Ain=np.array([[3,2],[1,3.]])
#Xin=np.array([[5,6],[9,10.]])

lcall=Ain@Xin

Nk=3
## NMF Initialization ============================
A0,X0=initnmf.init_random_direct(Nk,lcall)
A0=Ain
#X0=Xin
#X0=((pca.components_))

#A0,X0=initnmf.initpca(Nk,W,lcall,lamA)
#fac=np.sum(lcall)/np.sum(A0)/np.sum(X0)
#A0=A0*fac*0.1

trytag="LC401"
#regmode="L2"
regmode="L2-VRDet"
#regmode="L2-VRLD"
#regmode="Dual-L2"
Ntry=1000
lamA=10**-1
lamX=10**-1
epsilon=1.e-6
filename="test"
A,X,logmetric=runnmf.QP_NMF(regmode,Ntry,lcall,A0,X0,lamA,lamX,epsilon,filename,NtryAPGX=300,NtryAPGA=1,eta=1.e-6,endc=0.0,Lipfaca=1.e-3,Lipfacx=1.0)

print(np.sum((lcall-A@X)**2))
print(np.sum(lcall**2))

#print(A)
#print(lcall)
np.savez(filename,A,X)




