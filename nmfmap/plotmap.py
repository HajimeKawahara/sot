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
    fig=plt.figure(figsize=(10,7))
    s=0
    minval=np.min(resall)-1
    plt.plot(resall[s:,0]-minval,label="$||D-WAX||_F^2/2+R(A,X)$")
    plt.plot(resall[s:,1]-minval,label="$||D-WAX||_F^2/2$")
    plt.plot(resall[s:,2]-minval,label="$R(A)$")
    plt.plot(resall[s:,3]-minval,label="$R(X)$")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Interation #")
    plt.ylabel("Cost terms - mininum + 1")
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
    cc=plt.cm.viridis
    fontsize=18
    matplotlib.rcParams.update({'font.size':fontsize})
    hp.mollview(A[:,0], title="Component 0",flip="geo",cmap=cc)#,min=0,max=1)
    
    plt.savefig("C0.pdf", bbox_inches="tight", pad_inches=0.0)
    
    hp.mollview(A[:,1], title="Component 1",flip="geo",cmap=cc)#,min=0,max=1)
    plt.savefig("C1.pdf", bbox_inches="tight", pad_inches=0.0)
    
    hp.mollview(A[:,2], title="Component 2",flip="geo",cmap=cc)#,min=0,max=1)
    plt.savefig("C2.pdf", bbox_inches="tight", pad_inches=0.0)
    
    try:
        hp.mollview(A[:,3], title="Component 3",flip="geo",cmap=cc)#,min=0,max=1)
        plt.savefig("C3.pdf", bbox_inches="tight", pad_inches=0.0)
    except:
        print("No 4th comp")
        #hp.mollview(A[:,0]+A[:,1], title="0+1",flip="geo",cmap=plt.cm.jet)

def plref(X,bands):
    cloud,cloud_ice,snow_fine,snow_granular,snow_med,soil,veg,ice,water,clear_sky=io_refdata.read_refdata("/home/kawahara/exomap/sot/data/refdata")
    fig= plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    nnl=1#len(np.median(bands,axis=1))
    u,val,normvveg=norm(veg)
    ax.plot(u,val,c="black",lw=2,label="vegitation (deciduous)")
    u,val,normvsoil=norm(soil)
    ax.plot(u,val,c="gray",lw=1,label="soil")
    u,val,normvwater=norm(water)
    ax.plot(u,val,c="gray",ls="-.",label="water")
    plt.xlim(0.4,0.9)
    fac=1.0
    mband=np.median(bands,axis=1)
    dband=mband[1]-mband[0]
    fac0=fac/np.sum(X[0,:])/dband/normvveg
    fac1=fac/np.sum(X[1,:])/dband/normvwater
    fac2=fac/np.sum(X[2,:])/dband/normvsoil
    #fac3=fac/np.sum(X[3,:])/dband
    
    plt.plot(np.median(bands,axis=1),X[0,:]*fac0,"o",label="Component 0",color="C0")
    plt.plot(np.median(bands,axis=1),X[1,:]*fac1,"s",label="Component 1",color="C1")
    plt.plot(np.median(bands,axis=1),X[2,:]*fac2,"^",label="Component 2",color="C2")
    
    try:
        plt.plot(np.median(bands,axis=1),X[3,:]*fac3,"^",label="Component 3",color="C3")
        plt.plot(np.median(bands,axis=1),X[3,:]*fac3,color="C3")
        
    except:
        print("No 4th comp")
        
    plt.plot(np.median(bands,axis=1),X[0,:]*fac0,color="C0")
    plt.plot(np.median(bands,axis=1),X[1,:]*fac1,color="C1")
    plt.plot(np.median(bands,axis=1),X[2,:]*fac2,color="C2")
    
    plt.tick_params(labelsize=16)
    plt.ylabel("Reflection Spectra",fontsize=16)
    plt.xlabel("wavelength [micron]",fontsize=16)
    plt.legend(fontsize=13)
    plt.title("Unconstrained")
    plt.savefig("ref.pdf", bbox_inches="tight", pad_inches=0.0)

def classmap(A):
    Aclass=np.argmax(A,axis=1)
    Aclass[Aclass==0]=70
    Aclass[Aclass==1]=100
    Aclass[Aclass==2]=0
    #Aclass[Aclass==3]=30
    
    hp.mollview(Aclass, title="Retrieved",flip="geo",cmap=plt.cm.Greys,max=100)
    plt.savefig("retrieved.pdf", bbox_inches="tight", pad_inches=0.0)

def classmap_color(A):
    
   Nj=np.shape(A)[0]
   tip=0.1
   indx=np.array(range(0,Nj))

   #small norm filter
   Aabs=np.sqrt(np.sum(A**2,axis=1))
   crit=np.mean(Aabs)*0.15
   mask=Aabs<crit
   fac=0.75
   Anorm=A.T/np.sum(A,axis=1)*fac
   Anorm=Anorm.T
   Anorm=np.array([Anorm[:,2],Anorm[:,0],Anorm[:,1]]).T

   #fill value for small norm filter
   Anorm[mask]=np.sqrt(1.0/3.0)
   cmap = matplotlib.colors.ListedColormap(Anorm)
   hp.mollview(indx, title="Classification Map",flip="geo",cmap=cmap,min=0-tip,max=Nj-tip)
   plt.savefig("class.pdf", bbox_inches="tight", pad_inches=0.0)


def inmap():
    dataclass=np.load("/home/kawahara/exomap/sot/data/cmap3class.npz")
    cmapans=dataclass["arr_0"]
    hp.mollview(cmapans, title="Input",flip="geo",cmap=plt.cm.Greys_r)
    plt.savefig("input.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.show()

if __name__=='__main__':
    import sys
    #    axfile="npz/T116/T116_L2-VRLD_A-2.0X4.0j99000.npz"
    axfile=sys.argv[1]

    A,X,resall=read_data.readax(axfile)
    bands=read_data.getband()
    
#    plot_resall(resall)    
#    moll(A)
    plref(X,bands)
    classmap_color(A)
    plt.show()
    #    inmap()




