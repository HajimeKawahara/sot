import io_surface_type 
import io_refdata
import toymap
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import matplotlib
import mrsa
import read_data
import get_axfiles
import plotmap as pm
import diffclass
def plref_each(X,bands,ls,lab):
    cloud,cloud_ice,snow_fine,snow_granular,snow_med,soil,veg,ice,water,clear_sky=io_refdata.read_refdata("/home/kawahara/exomap/sot/data/refdata")
    nnl=1#len(np.median(bands,axis=1))
    u,val,normvveg=pm.norm(veg)
    u,val,normvsoil=pm.norm(soil)
    u,val,normvwater=pm.norm(water)
    plt.xlim(0.4,0.9)
    fac=1.0
    mband=np.median(bands,axis=1)
    dband=mband[1]-mband[0]
    fac0=fac/np.sum(X[0,:])/dband/normvveg
    fac1=fac/np.sum(X[1,:])/dband/normvsoil    
    fac2=fac/np.sum(X[2,:])/dband/normvwater
    plt.plot(np.median(bands,axis=1),X[0,:]*fac0,color="C2",lw=2,ls=ls)
    plt.plot(np.median(bands,axis=1),X[1,:]*fac1,color="C1",lw=2,ls=ls)
    plt.plot(np.median(bands,axis=1),X[2,:]*fac2,color="C0",lw=2,ls=ls,label=lab)
    
    plt.tick_params(labelsize=16)

def plot_regx(axfiles):
    NN=512*10
    lcmean=105.70907195459881

    mrarr=[]
    likarr=[]
    detx=[]
    detxn=[]
    absa=[]
    for i,axfile in enumerate(axfiles):
        A,X,resall=read_data.readax(axfile)
        mmrsa=mrsa.mrsa_meanX(X)
        AFnorm=np.sqrt(np.sum(A*A))
        mrarr.append(mmrsa)
        likarr.append(resall[-1][1])
        absa.append(resall[-1][2])
        Xn=np.copy(X)
        for k in range(0,np.shape(X)[0]):
            Xn[k,:]=X[k,:]/np.sum(X[k,:])
        detxn.append(np.linalg.det(np.dot(Xn,Xn.T)))#/10**lam[i])
        detx.append(np.linalg.det(np.dot(X,X.T)))#/10**lam[i])
        
        print(lam[i],resall[-1][1],AFnorm,mmrsa)
    likarr=np.array(likarr)
    absa=np.array(absa)
    detx=np.array(detx)
    detxn=np.array(detxn)

    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})
    fig=plt.figure(figsize=(7,7.5))
    ax=fig.add_subplot(311)
    ax.plot(10**lam,np.sqrt(likarr/NN)/lcmean,"o",color="C0")
    ax.plot(10**lam,np.sqrt(likarr/NN)/lcmean,color="C0")
    plt.ylabel("mean residual")
    plt.xscale("log")
    ax=fig.add_subplot(312)    
    plt.xscale("log")
#    plt.yscale("log")
#    ax.plot(10**lam,(absa),"o",)
    ax.plot(10**lam,(detxn),"o",color="C0")
    ax.plot(10**lam,(detxn),color="C0")
#    ax.plot(10**lam,(detxn)*1000,color="C0")

    plt.ylabel("$\det{(\hat{X} \hat{X}^T)}$")

#    plt.ylabel("$||D - W A X||_F^2$")
    ax=fig.add_subplot(313)
    ax.plot(10**lam,mrarr,"o",color="C0")
    ax.plot(10**lam,mrarr,color="C0")
    plt.xscale("log")
    plt.ylabel("$\overline{MRSA}$")
    plt.xlabel("$\lambda_X$")
    plt.savefig("regx.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()

def plotrefdepx(axfiles,lam):
    bands=read_data.getband()
    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})
    fig= plt.figure(figsize=(7,3))
    ax = fig.add_subplot(111)
    lss=["dashed","dashed","dashed","solid","solid","dotted","dotted"]
    for i in [0,4,6]:
        axfile=axfiles[i]
        A,X,resall=read_data.readax(axfile)
        print(lam[i])
        plref_each(X,bands,lss[i],"$\lambda_X=10^{"+str((lam[i]))+"}$")
    plt.ylabel("Unmixed Spectra",fontsize=16)
    plt.xlabel("wavelength [micron]",fontsize=16)
    plt.legend(fontsize=13)
    plt.savefig("refdep.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()

def plot_rega(axfiles):
    mrarr=[]
    mrsaarr=[]
    likarr=[]
    for i,axfile in enumerate(axfiles):
        A,X,resall=read_data.readax(axfile)
        Aclass=diffclass.mclassmap(A)
        mr=diffclass.classdif(Aclass)
        AFnorm=np.sqrt(np.sum(A*A))
        mmrsa=mrsa.mrsa_meanX(X)
        mrsaarr.append(mmrsa)
        mrarr.append(mr)
        likarr.append(resall[-1][1])
        print(lam[i],resall[-1][1],AFnorm,mr,mmrsa)

    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})
    likarr=np.array(likarr)
    NN=512*10
    lcmean=105.70907195459881
    fig=plt.figure(figsize=(7,7.5))
    ax=fig.add_subplot(311)
    ax.plot(10**lam,np.sqrt(likarr/NN)/lcmean,"o",color="C0")
    ax.plot(10**lam,np.sqrt(likarr/NN)/lcmean,color="C0")
    plt.xscale("log")
#    plt.yscale("log")

#    plt.ylabel("$||D - W A X||_F^2$")
    plt.ylabel("mean residual")
    ax=fig.add_subplot(312)
    ax.plot(10**lam,mrsaarr,"o",color="C0")
    ax.plot(10**lam,mrsaarr,color="C0")
    plt.xscale("log")
    plt.ylabel("$\overline{MRSA}$")
    
    ax=fig.add_subplot(313)
    ax.plot(10**lam,mrarr,"o",color="C0")
    ax.plot(10**lam,mrarr,color="C0")
    plt.xscale("log")
    plt.ylabel("CPR")
    plt.xlabel("$\lambda_A$")
    plt.savefig("rega.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()

if __name__=='__main__':
    import sys
#    axfiles=sys.argv[1:]
    axfiles,lam=get_axfiles.get_axfiles_X()
#    plotrefdepx(axfiles,lam)
    plot_regx(axfiles)
#    plt.show()
#    axfiles,lam=get_axfiles.get_axfiles_A()
#    plot_rega(axfiles)
