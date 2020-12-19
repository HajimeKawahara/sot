import io_surface_type 
import io_refdata
import toymap
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import matplotlib
import read_data
import mrsa
import plotmap as pm
import get_axfiles
import metric
import plotmap as pm

def plotrefdirect(axfiles,lam):
    bands=read_data.getband()
    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})
    fig= plt.figure(figsize=(8,5.5))
    ax = fig.add_subplot(111)
    cloud,cloud_ice,snow_fine,snow_granular,snow_med,soil,veg,ice,water,clear_sky=io_refdata.read_refdata("/home/kawahara/exomap/sot/data/refdata")
    nnl=1#len(np.median(bands,axis=1))
    u,val,normvveg=pm.norm(veg)
#    ax.plot(u,val,c="gray",lw=2,label="vegitation (deciduous)")
    ax.plot(u,val,c="gray",lw=2)    
    u,val,normvsoil=pm.norm(soil)
    ax.plot(u,val,c="gray",lw=2,ls="dashed")
    #    ax.plot(u,val,c="gray",lw=2,ls="dashed",label="soil")
    u,val,normvwater=pm.norm(water)
    ax.plot(u,val,c="gray",lw=2,ls="-.")

#    ax.plot(u,val,c="gray",lw=2,ls="-.",label="water")


    lss=["dashed","solid","dotted","dashdot"]
    sy=[False,True,False]
    for i in range(0,3):
        axfile=axfiles[i]
        A,X,resall=read_data.readax(axfile)
        mmrsa=mrsa.mrsa_meanX(X)
        print(mmrsa)
        print(lam[i])
        
        metric.plref_each(X,bands,lss[i],"$\lambda_X=10^{"+str(int(lam[i]))+"}$",symbol=sy[i])
    plt.ylim(0,0.6)
    plt.ylabel("Unmixed Spectra",fontsize=16)
    plt.xlabel("wavelength [micron]",fontsize=16)
    plt.legend(fontsize=13)
    plt.savefig("refdirect.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()

    
if __name__=='__main__':
    import sys
    axfiles,lam=get_axfiles.get_axfiles_directX()
    print(axfiles)
    plotrefdirect(axfiles,lam)
