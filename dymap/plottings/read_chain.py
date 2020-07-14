import emcee
import numpy as np
import matplotlib.pyplot as plt
dat=np.load("flat_sample_dyRBFspin_gdp.npz",allow_pickle=True)
#dat=np.load("samples/test62/flat_sample_dyRBFspin.npz",allow_pickle=True)
flat_samples=dat["arr_0"]
inc,Thetaeq,zeta,Pspin,Porb,obst=dat["arr_3"]
labels=["zeta","Thetaeq","gamma","alpha","tau","pspin"]
nsample,nwalk=(np.shape(flat_samples))
burnin=1000
fig=plt.figure()
for i in range(0,nwalk):
    ax=fig.add_subplot(nwalk+1,1,i+1)
    ax.plot(flat_samples[:,i])
    plt.axvline(burnin,color="red")
plt.show()

import corner
fig = corner.corner(flat_samples[burnin:,:], labels=labels, truths=[zeta,Thetaeq,None,None,None,Pspin])
plt.show()
