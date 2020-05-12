import numpy as np
import pylab
import matplotlib.pyplot as plt


def PDF2D(parray,xlist,ylist,pcont,clevin="auto"):
    def findval(p):
        ip=np.searchsorted(csum,p)
        return sortflat[ip]

    #2D PDF color map + contours
    #pcont : contour values
    #clevin : User-defined contour labels
    
    #thetaeqlist/np.pi*180.0,zetalist/np.pi*180.0
    #pcont=[0.995,0.9,0.5]
    #["99.5%","90%","50%"]
    pcont=np.sort(pcont)[::-1]
    parray=parray/np.sum(parray)
    flat=parray.flatten()
    sortflat=np.sort(flat)[::-1]
    csum=np.cumsum(sortflat)
    X, Y=pylab.meshgrid(xlist,ylist)
    clist=[]
    clev=[]
    for i,val in enumerate(pcont):
        clist.append(findval(val))
        clev.append(str(val*100)+'%')
    CS=plt.contour(X, Y,parray,clist,colors='white', linestyles='solid')
    if clevin == "auto":
        CS.levels=clev
        plt.clabel(CS, CS.levels, inline=True, fontsize=13)
    elif clevin is not None:
        CS.levels=clevin
        plt.clabel(CS, CS.levels, inline=True, fontsize=13)
        
    plt.imshow(parray,cmap="plasma",extent=[xlist[0],xlist[-1],ylist[-1],ylist[0]])
    plt.gca().invert_yaxis()

               
               #               extent=[(thetaeqlist[0])/np.pi*180.0,(thetaeqlist[-1])/np.pi*180.0,(zetalist[-1])/np.pi*180.0,(zetalist[0])/np.pi*180.0])
               #    plt.plot(Thetaeq/np.pi*180.0,zeta/np.pi*180.0,"*",color="white",label="Input",markersize=12)
#    plt.legend()
#    plt.xlabel("$\\Theta_\\mathrm{eq}$ (degree)",fontsize=16)
#    plt.ylabel("$\\zeta$ (degree)",fontsize=16)
#    plt.tick_params(labelsize=13)
#    plt.savefig("axtexpand.pdf", bbox_inches="tight", pad_inches=0.0)
