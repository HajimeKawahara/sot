import io_surface_type 
import io_refdata
import toymap
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import matplotlib
from sklearn.decomposition import PCA

#dat=np.load("npznew/DetAX_a-2.0x1.0_try100000j32000.npz")
dat=np.load("npznew/L2AX_a-2.0x2.0_try100000j19000.npz")
#dat=np.load("npznew/uncAX_a-2.0x-inf_try100000j19000.npz")

A=dat["arr_0"]
X=dat["arr_1"]

lcall=np.load("lcall.npz")["arr_0"]
pca = PCA(n_components=2)                                                                                         
pca.fit(lcall) #x(k,l)                                                                              
Xp=((pca.components_))
print()
#for k in range(0,len(Xp)):
#    uk=Xp[k,:]
pc1=Xp[0,:]
pc2=Xp[1,:]
npix=3072
xpc1=np.dot(X,pc1)
xpc2=np.dot(X,pc2)

lcpc1=np.dot(lcall/500,pc1)
lcpc2=np.dot(lcall/500,pc2)


print(np.concatenate([xpc1,xpc1]))

plt.figure()
plt.plot(xpc1,xpc2,"o")
plt.plot(np.concatenate([xpc1,xpc1]),np.concatenate([xpc2,xpc2]))
plt.plot(lcpc1,lcpc2,".")

plt.show()
