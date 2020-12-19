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

def plotseqmap(A,frames,tag="map",title="",zero=True,contrast=1.0):
    if zero:
        A[A==0.0]=None
    if np.shape(np.shape(A))[0]==2:
        for i in tqdm(frames):
            hp.mollview(A[i,:], title=title,flip="geo",cmap=plt.cm.bone,min=0.0,max=contrast)
            plt.savefig("png/"+tag+str(i)+".png")
            plt.savefig("pdf/"+tag+str(i)+".pdf")

            plt.close()
    elif np.shape(np.shape(A))[0]==3:
        import plotmap
        Nk=np.shape(A)[2]
        for i in tqdm(frames):
            tagc=tag+"c"+str(i)
            plotmap.classmap_color(A[i,:,:],title=title,theme="3c",pdfname="pdf/"+tagc+".pdf",pngname="png/"+tagc+".png")
            for k in range(0,Nk):
                hp.mollview(A[i,:,k], title=title,flip="geo",cmap=plt.cm.bone)
                plt.savefig("png/"+tag+str(i)+"_"+str(k)+".png")
                plt.savefig("pdf/"+tag+str(i)+"_"+str(k)+".pdf")
                plt.close()
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot dynamic map')
    parser.add_argument('-f', nargs=1, default=["test.npz"], help='npz file', type=str)
    args = parser.parse_args()

    dat=np.load(args.f[0])
    A=dat["arr_0"]
    X=dat["arr_1"]
    resall=dat["arr_2"]

    Nt=np.shape(A)[0]
    frames=[0,int(Nt/2),Nt-1]

    B=np.copy(A[:,:,1])
    A[:,:,1]=np.copy(A[:,:,2])
    A[:,:,2]=B

    plotseqmap(A,frames)
