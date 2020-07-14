import emcee
import numpy as np
import matplotlib.pyplot as plt

# LOAD sampled parameters


fsample="../samples/test531/flat_sample_dscovrRBF.npz"
burnin=500

dat=np.load(fsample,allow_pickle=True)
flat_samples=dat["arr_0"]

nsample,nwalk=(np.shape(flat_samples))
fig=plt.figure()
for i in range(0,nwalk):
    ax=fig.add_subplot(nwalk+1,1,i+1)
    ax.plot(flat_samples[:,i])
    plt.axvline(burnin,color="red")
plt.show()


Ns,Npar=np.shape(flat_samples)
Nsample=Ns

print("MEDIAN g,a,t,s")
print(np.median(flat_samples,axis=0))
flat_samples_corner=np.copy(flat_samples)
flat_samples_corner[:,0]=flat_samples[:,0]/np.pi*180 #rad2deg

W=dat["arr_1"]
lc=dat["arr_2"]

########################################
#POSTERIOR
########################################

labels=["$\\gamma$ (deg)","$\\alpha$","$\\tau$ (d)","$\\sigma$"]
tag="RBF531"
import corner
fig = corner.corner(flat_samples_corner[burnin:,:], labels=labels, truths=[None,None,None,None],truth_color="C1")
plt.savefig("corner_dscovr"+tag+".png")
plt.savefig("corner_dscovr"+tag+".pdf")
plt.show()

