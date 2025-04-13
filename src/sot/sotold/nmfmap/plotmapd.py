import io_surface_type 
import io_refdata
import toymap
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import matplotlib
import mrsa
import read_data

def norm(arr):
    mask=(arr[:,0]>0.4)*(arr[:,0]<0.9)
    from scipy import interpolate
    f = interpolate.interp1d(arr[:,0], arr[:,1], kind='linear')
    ls,le,ndiv=0.4,0.9,100
    u=np.linspace(ls,le,ndiv)
    du=(le-ls)/ndiv
    val=f(u)
    normv=1.0/np.sum(val)/du
    val=val
    return u,val,normv

def plot_resall(resall):
    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})
    fig=plt.figure(figsize=(9,9))
    ax=fig.add_subplot(211)
    s=0
    minval=0.0
    plt.plot(resall[s:,0]-minval,label="$||D-WAX||_F^2/2+R(A,X)$",rasterized=True)
    plt.plot(resall[s:,1]-minval,label="$||D-WAX||_F^2/2$",rasterized=True,ls="dashed")
    plt.plot(resall[s:,2]-minval,label="$R(A)=\lambda_A ||A||_F^2/2$",rasterized=True,ls="dotted")
    plt.plot(resall[s:,3]-minval,label="$R(X)=\lambda_X \det{(X X^T)}/2$",rasterized=True,ls="dashdot")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Cost terms")

    ax=fig.add_subplot(212)
    resdiff=-resall[1:,:]+resall[0:-1,:]
    plt.plot(resdiff[s:,0]-minval,label="$||D-WAX||_F^2/2+R(A,X)$",rasterized=True)
    plt.axhline(1.e-5,color="C4",ls="dashed")
#    plt.plot(resdiff[s:,1]-minval,label="$||D-WAX||_F^2/2$")
#    plt.plot(resdiff[s:,2]-minval,label="$R(A)$")
#    plt.plot(resdiff[s:,3]-minval,label="$R(X)$")
#    plt.legend()

    plt.yscale("log")
    plt.xscale("log")
    
    plt.xlabel("Iteration #")
    plt.ylabel("Difference")
    plt.savefig("resevo.pdf", bbox_inches="tight", pad_inches=0.0)

    plt.show()

def plot_resdiff(resall):
    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})
    fig=plt.figure(figsize=(10,7))
    s=0
    minval=0.0
    resdiff=-resall[1:,:]+resall[0:-1,:]
    plt.plot(resdiff[s:,0]-minval,label="$||D-WAX||_F^2/2+R(A,X)$")
#    plt.plot(resdiff[s:,1]-minval,label="$||D-WAX||_F^2/2$")
#    plt.plot(resdiff[s:,2]-minval,label="$R(A)$")
#    plt.plot(resdiff[s:,3]-minval,label="$R(X)$")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel("Interation #")
    plt.ylabel("Cost terms")
    plt.show()


def load_template():
    ## load class map
    dataclass=np.load("/home/kawahara/exomap/sot/data/cmap3class.npz")
    cmap=dataclass["arr_0"]
    npix=len(cmap)
    nclass=(len(np.unique(cmap)))
    nside=hp.npix2nside(npix)
    vals=dataclass["arr_1"]
    valexp=dataclass["arr_2"]
    print("Nclass=",nclass)
    
    #mean albedo between waves and wavee
    #bands=[[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8],[0.8,0.9]]#,[0.9,1.0]]
    
    refsurfaces=[water,soil,veg]
    malbedo=io_surface_type.set_meanalbedo(0.8,0.9,refsurfaces,clear_sky)
    mmap,Ain,Xin=toymap.make_multiband_map(cmap,refsurfaces,clear_sky,vals,bands)
    ave_band=np.mean(np.array(bands),axis=1)
    return 


def moll(A):
    fc=0.9
    bc=1.0
    cc=plt.cm.viridis
    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})
    hp.mollview(A[:,0], title="Component 0",flip="geo",cmap=cc,min=np.min(A[:,0])*bc,max=np.max(A[:,0])*fc)
    
    plt.savefig("C0.pdf", bbox_inches="tight", pad_inches=0.0)
    
    hp.mollview(A[:,1], title="Component 1",flip="geo",cmap=cc,min=np.min(A[:,1])*bc,max=np.max(A[:,1])*fc)#,min=0,max=1)
    plt.savefig("C1.pdf", bbox_inches="tight", pad_inches=0.0)
    
    hp.mollview(A[:,2], title="Component 2",flip="geo",cmap=cc,min=np.min(A[:,2])*bc,max=np.max(A[:,2])*fc)#,min=0,max=1)
    plt.savefig("C2.pdf", bbox_inches="tight", pad_inches=0.0)
    
    try:
        hp.mollview(A[:,3], title="Component 3",flip="geo",cmap=cc)#,min=0,max=1)
        plt.savefig("C3.pdf", bbox_inches="tight", pad_inches=0.0)
    except:
        print("No 4th comp")
        #hp.mollview(A[:,0]+A[:,1], title="0+1",flip="geo",cmap=plt.cm.jet)

def plref(X,bands,theme,title="",oxlab=False):
    cloud,cloud_ice,snow_fine,snow_granular,snow_med,soil,veg,ice,water,clear_sky=io_refdata.read_refdata("/home/kawahara/exomap/sot/data/refdata")
    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})

    fig= plt.figure(figsize=(8,5.5))
    ax = fig.add_subplot(111)
    nnl=1#len(np.median(bands,axis=1))
    if theme=="3c":
        u,val,normvveg=norm(veg)
        ax.plot(u,val,c="gray",lw=2,label="vegitation (deciduous)")
        u,val,normvsoil=norm(soil)
        ax.plot(u,val,c="gray",lw=2,ls="dashed",label="soil")
        u,val,normvwater=norm(water)
        ax.plot(u,val,c="gray",lw=2,ls="-.",label="water")
    else:
        normvveg=1.0
        normvsoil=1.0
        normvwater=1.0
    fac=1.0
    mband=np.median(bands,axis=1)
    dband=mband[1]-mband[0]
    if theme=="3c":
        plt.xlim(0.4,0.9)

        fac0=fac/np.sum(X[0,:])/dband/normvveg
        fac1=fac/np.sum(X[1,:])/dband/normvsoil
        fac2=fac/np.sum(X[2,:])/dband/normvwater
    else:
        plt.xlim(0.3,0.9)

        try:
            fac0=fac/np.sum(X[0,:])/dband
            fac1=fac/np.sum(X[1,:])/dband
            fac2=fac/np.sum(X[2,:])/dband
            fac3=fac/np.sum(X[3,:])/dband
        except:
            print("No 4th comp")
            
    if theme=="3c":
        plt.plot(np.median(bands,axis=1),X[0,:]*fac0,"o",label="Component 0",color="C2")
        plt.plot(np.median(bands,axis=1),X[1,:]*fac1,"s",label="Component 1",color="C1")
        plt.plot(np.median(bands,axis=1),X[2,:]*fac2,"^",label="Component 2",color="C0")
    
        try:
            plt.plot(np.median(bands,axis=1),X[3,:]*fac3,"^",label="Component 3",color="C3")
            plt.plot(np.median(bands,axis=1),X[3,:]*fac3,color="C3")        
        except:
            print("No 4th comp")
        
        plt.plot(np.median(bands,axis=1),X[0,:]*fac0,color="C2",lw=2)
        plt.plot(np.median(bands,axis=1),X[1,:]*fac1,color="C1",lw=2)
        plt.plot(np.median(bands,axis=1),X[2,:]*fac2,color="C0",lw=2)
    elif theme=="dscovr":
        #c,o,v,l
        plt.plot(np.median(bands,axis=1),X[0,:]*fac0,"o",label="Component 0",color="gray")
        plt.plot(np.median(bands,axis=1),X[1,:]*fac1,"s",label="Component 1",color="C0")
        plt.plot(np.median(bands,axis=1),X[2,:]*fac2,"^",label="Component 2",color="C2")
        plt.plot(np.median(bands,axis=1),X[3,:]*fac3,"*",label="Component 3",color="C3")

        plt.plot(np.median(bands,axis=1),X[0,:]*fac0,color="gray",lw=2)
        plt.plot(np.median(bands,axis=1),X[1,:]*fac1,color="C0",lw=2)
        plt.plot(np.median(bands,axis=1),X[2,:]*fac2,color="C2",lw=2)
        plt.plot(np.median(bands,axis=1),X[3,:]*fac3,color="C3",lw=2)

    if oxlab:
        plt.axvline(0.688,color="blue",alpha=0.2,lw=5)
        plt.axvline(0.764,color="blue",alpha=0.2,lw=5)
#        plt.text("")
    plt.tick_params(labelsize=16)
    plt.ylabel("Reflection Spectra",fontsize=16)
    plt.xlabel("wavelength [micron]",fontsize=16)
    plt.legend(fontsize=12)
    plt.title(title)
    plt.savefig("ref.pdf", bbox_inches="tight", pad_inches=0.0)

def classmap(A,title="",theme="3c"):
    Aclass=np.argmax(A,axis=1)
    if theme=="3c":
        Aclass[Aclass==0]=70
        Aclass[Aclass==1]=0
        Aclass[Aclass==2]=100
    if theme=="dscovr":
        Aclass[Aclass==0]=0
        Aclass[Aclass==1]=100
        Aclass[Aclass==2]=70
        Aclass[Aclass==3]=50

    #Aclass[Aclass==3]=30
    Aabs=np.sqrt(np.sum(A**2,axis=1))
    crit=np.mean(Aabs)*0.15
    mask=Aabs<crit
    Aclass[mask]=30
    
    hp.mollview(Aclass, title="Classification "+title,flip="geo",cmap=plt.cm.Greys,max=100,cbar=None)
    plt.savefig("retrieved.pdf", bbox_inches="tight", pad_inches=0.0)

def classmap_color(A,title="",theme="b"):
    
   Nj=np.shape(A)[0]
   tip=0.1
   indx=np.array(range(0,Nj))

   #small norm filter
   Aabs=np.sqrt(np.sum(A**2,axis=1))
   crit=np.mean(Aabs)*0.015
   mask=Aabs<crit
   fac=0.75
#   fac=1.0
   gg=0.9
   rr=0.95
   bb=0.0
   if theme=="3c":
       B=np.copy(A[:,1])
       A[:,1]=np.copy(A[:,2])
       A[:,2]=B
       rot=np.array([[gg,0,np.sqrt(1.0-gg*gg)],[0,1,0],[np.sqrt(1.0-rr*rr-bb*bb),bb,rr]])                   
       A=np.dot(A,rot)
       Anorm=A.T/np.sum(A,axis=1)*fac
       Anorm=Anorm.T
       bright=np.array([[1.0,0.1,0.1],[0.1,1.0,0.1],[0.2,0.2,1.0]])
       Anorm=np.dot(Anorm,bright)
       Anorm=np.array([Anorm[:,2],Anorm[:,0],Anorm[:,1]]).T

   elif theme=="dscovr":
       #c,o,v,l
       Acl=A[:,0]
       Acl=Acl/np.max(Acl)*5.0+1.0
       #rot=np.array([[1.0,1.0,1.0],[0,1,0],[gg,0,np.sqrt(1.0-gg*gg)],[np.sqrt(1.0-rr*rr-bb*bb),bb,rr]])
       rot=np.array([[1.0,1.0,1.0],[0,1,0],[np.sqrt(1.0-rr*rr-bb*bb),bb,rr],[gg,0,np.sqrt(1.0-gg*gg)]])
       
       A=np.dot(A,rot)
       Anorm=A.T/np.sum(A,axis=1)*fac
       Anorm=Anorm.T
       bright=np.array([[1.0,0.1,0.1],[0.1,1.0,0.1],[0.2,0.2,1.0]])
       #       bright=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]])

       Anorm=np.dot(Anorm,bright)
       #       Anorm[cmask,:]=Anorm[cmask,:]*2.0
       B=Anorm[:,2]*Acl
       C=Anorm[:,0]*Acl
       D=Anorm[:,1]*Acl
       B[B>1.0]=1.0
       C[C>1.0]=1.0
       D[D>1.0]=1.0
       
       Anorm=np.array([B,C,D]).T
       

   #fill value for small norm filter
   Anorm[mask]=np.sqrt(1.0/3.0)*0.1
   cmap = matplotlib.colors.ListedColormap(Anorm)
   hp.mollview(indx, title="Color composite "+title,flip="geo",cmap=cmap,min=0-tip,max=Nj-tip,cbar=None)
   plt.savefig("class.pdf", bbox_inches="tight", pad_inches=0.0)
   plt.savefig("class.png", bbox_inches="tight", pad_inches=0.0)


def inmap():
    dataclass=np.load("/home/kawahara/exomap/sot/data/cmap3class.npz")
    cmapans=dataclass["arr_0"]
    hp.mollview(cmapans, title="Input",flip="geo",cmap=plt.cm.Greys_r,cbar=None)
    plt.savefig("input.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()

def getpred(A,X):
    W=np.load("w.npz")["arr_0"]
    WA=np.dot(W,A)
    pred=np.dot(WA,X)
    lcall=np.load("lcall.npz")["arr_0"]
    return lcall, pred

def showpred(A,X):
    lcall,pred=getpred(A,X)
    fig=plt.figure()
    for i in range(0,np.shape(lcall)[1]):
        ax=fig.add_subplot(np.shape(lcall)[1],2,2*i+1)
        ax.plot((lcall[:,i]-pred[:,i])/lcall[:,i]*100.0,".")
        ax=fig.add_subplot(np.shape(lcall)[1],2,2*i+2)
        ax.plot((lcall[:,i]),".")
        ax.plot((pred[:,i]))

#        ax.plot(pred[:,i])
    plt.savefig("lc.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()
    
if __name__=='__main__':
    import sys
    #    axfile="npz/T116/T116_L2-VRLD_A-2.0X4.0j99000.npz"
#    theme="dscovr"
    theme="dscovr"
    axfile=sys.argv[1]
    Ns=sys.argv[2]
    A,X,resall=read_data.readax(axfile)
    NN=2435*7.

    print(np.sqrt(resall[-1,:]/NN))
#    NN=1
    lcmean=107.40646786191814
    lcsig=lcmean*0.03
    Nd=2435*7
    print("Ln L=",resall[-1,:]/(2*lcsig*lcsig))
    print("AIC=",2*resall[-1,:]/(2*lcsig*lcsig)+2*(3072*3))
    #bands=read_data.getband()
    bands=[[0.388,0.388],[0.443,0.443],[0.552,0.552],[0.680,0.680],[0.688,0.688],[0.764,0.764],[0.779,0.779]] #DSCOVR
    
#    showpred(A,X)
#    sys.exit()

    
    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})
    title=""
    oxlab=True
    plot_resall(resall)    
#    plot_resdiff(resall)    
    moll(A)
    plref(X,bands,theme,title,oxlab)
    classmap(A)
    classmap_color(A,title,theme=theme)
    plt.show()
    #inmap()



#    bands=[0.388,0.443,0.552,0.680,0.688,0.764,0.779]]

