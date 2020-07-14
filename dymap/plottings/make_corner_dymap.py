import emcee
import numpy as np
import matplotlib.pyplot as plt

# LOAD sampled parameters
fsample="../samples/test62/flat_sample_dyRBFspin.npz"
tag="RBFspin"
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

print(np.shape(flat_samples))
flat_samples_corner=np.copy(flat_samples)
flat_samples_corner[:,0:3]=flat_samples[:,0:3]/np.pi*180 #rad2deg
#    flat_samples_corner[:,5]=(flat_samples_corner[:,5])

W=dat["arr_1"]
lc=dat["arr_2"]
inputgeo=dat["arr_3"]
inc,Thetaeq,zeta,Pspin,Porb,obst=inputgeo
worb=2*np.pi/Porb                                                          
Thetav=worb*obst

sigma=np.mean(lc)*0.01
########################################
#POSTERIOR
########################################

labels=["$\\zeta$ (deg)","$\\Theta_\\mathrm{eq}$ (deg)","$\\gamma$ (deg)","$\\alpha$","$\\tau$ (d)",""]
import corner
print(np.shape(flat_samples_corner),np.shape(labels))
fig = corner.corner(flat_samples_corner[burnin:,:], labels=labels, truths=[zeta/np.pi*180,Thetaeq/np.pi*180,None,None,None,Pspin],truth_color="C1")
ndim=np.shape(flat_samples_corner)[1]
axes = np.array(fig.axes).reshape((ndim, ndim))
ax=axes[4,5]
ax_pos = ax.get_position()
#    ax.text(0.5,0.5, "P(d)",fontsize=20)
ax.text(ax_pos.x1-0.1, ax_pos.y1-1.785, "$P_\\mathrm{spin}$(d)",fontsize=14)
ax=axes[4,0]
ax_pos = ax.get_position()
#    ax.text(0.5,0.5, "P(d)",fontsize=20)
ax.text(ax_pos.x1+12.0, ax_pos.y1, "$P_\\mathrm{spin}$(d)",fontsize=14,rotation=90)
greenarea=False
if greenarea:
    ax=axes[5,5]
    #    ax.axvline(Pspin+1.0/365/64)
    #    ax.axvline(Pspin-1.0/365/64)
    ax.fill_between([Pspin+1.0/365/64,Pspin-1.0/365/64],1.e3,color="green",alpha=0.5)
    ax=axes[1,0]
    ax.plot(zeta/np.pi*180,Thetaeq/np.pi*180,"s",color="red")
    for i in range(0,5):
        ax=axes[5,i]
        #        ax.axhline(Pspin+1.0/365/64)
        #        ax.axhline(Pspin-1.0/365/64)
        minv=np.min(flat_samples_corner[:,i])
        maxv=np.max(flat_samples_corner[:,i])
        ax.fill_between([minv,maxv],Pspin+1.0/365/64,Pspin-1.0/365/64,color="green",alpha=0.5)
        
plt.savefig("corner_dy"+tag+".png", bbox_inches="tight", pad_inches=0.0)
plt.savefig("corner_dy"+tag+".pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()
