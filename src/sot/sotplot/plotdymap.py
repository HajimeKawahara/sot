import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import healpy as hp

def plotpred(obst,lc,pred,pred_static):
    fig=plt.figure()
    ax=fig.add_subplot(211)
    plt.plot(obst,lc,".",label="data")
    plt.plot(obst,pred_static,label="fit",alpha=0.7,lw=0.75)
    plt.legend()
    plt.ylabel("Static SOT")
    ax=fig.add_subplot(212)
    plt.plot(obst,lc,".",label="data")
    plt.plot(obst,pred,label="fit",alpha=0.7,lw=0.75)
    plt.legend()
    plt.ylabel("Dynamic SOT")
    plt.xlabel("Day")
    plt.savefig("lc.pdf")
    plt.savefig("lc.png")

def bound_earth(boundary_data):
    dat=np.load(boundary_data)
    nside=dat["arr_0"]
    nbound=dat["arr_1"]
    theta,phi=hp.pixelfunc.pix2ang(nside,nbound)
    return theta,phi
    
def plotseqmap(A,frames,tag="map",title=None,zero=True,vmin=None,vmax=None,cmap=plt.cm.pink,Earth=False,boundary_data="/home/kawahara/exomap/sot/data/earth_boundary.npz",color="#66CC99"):
    if Earth:
        thetaE,phiE=bound_earth(boundary_data)
    
    if title is None or len(title) != len(frames):
        print("No title or mismatch")
        title=[]
        for i in frames:
            title.append("")
    if zero:
        A[A==0.0]=None
    if np.shape(np.shape(A))[0]==2:
        j=0
        for i in tqdm(frames):
            if vmin is None:
                hp.mollview(A[i,:], title=title[j],flip="geo",cmap=cmap)
                if Earth:
                    hp.projplot(thetaE, phiE,".",c=color) 

            else:
                hp.mollview(A[i,:], title=title[j],flip="geo",cmap=cmap,min=vmin,max=vmax)
                if Earth:
                    hp.projplot(thetaE, phiE,".",c=color,alpha=0.5) 

            plt.savefig("png/"+tag+str(i)+".png")
            plt.savefig("pdf/"+tag+str(i)+".pdf")
            plt.close()
            j=j+1

    elif np.shape(np.shape(A))[0]==3:
        import plotmap
        Nk=np.shape(A)[2]
        j=0
        for i in tqdm(frames):
            tagc=tag+"c"+str(i)
            plotmap.classmap_color(A[i,:,:],title=title[j],theme="3c",pdfname="pdf/"+tagc+".pdf",pngname="png/"+tagc+".png")
            for k in range(0,Nk):
                if vmin is None:
                    hp.mollview(A[i,:,k], title=title[j],flip="geo",cmap=cmap)
                else:
                    hp.mollview(A[i,:,k], title=title[j],flip="geo",cmap=cmap,min=vmin,max=vmax)
                plt.savefig("png/"+tag+str(i)+"_"+str(k)+".png")
                plt.savefig("pdf/"+tag+str(i)+"_"+str(k)+".pdf")
                plt.close()
            j=j+1
    else:
        print("It's not dynamic map! shape=",np.shape(np.shape(A))[0])
        
if __name__ == "__main__":
    import healpy as hp
    import pylab 
    import matplotlib.pyplot as plt
    import mvmap
    import matplotlib
    fontsize=16
    matplotlib.rcParams.update({'font.size':fontsize})

    mmap=hp.read_map("/home/kawahara/exomap/sot/data/mockalbedo16.fits")
    mask=(mmap>0.0)
    mmap[mask]=1.0
    Porb=365.242190402                                            
    Ni=1024
    obst=np.linspace(0.0,Porb,Ni)
    M=mvmap.rotating_map(mmap,obst,rotthetamax=np.pi/2.0)
    frames=[0,int(Ni/2),Ni-1] 
    plotseqmap(M,frames,"mapin",title=["0 day","182 day","364 day"],vmin=0.0,vmax=1.0,cmap=plt.cm.bone,zero=False)

