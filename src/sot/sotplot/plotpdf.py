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
    try:
        CS=plt.contour(X, Y,parray,clist,colors='white', linestyles='solid')
        if clevin == "auto":
            CS.levels=clev
            plt.clabel(CS, CS.levels, inline=True, fontsize=13)
        elif clevin is not None:
            CS.levels=clevin
            plt.clabel(CS, CS.levels, inline=True, fontsize=13)
    except:
        print("Contour unavailble.")
    plt.imshow(parray,cmap="plasma",extent=[xlist[0],xlist[-1],ylist[-1],ylist[0]])
    plt.gca().invert_yaxis()

