import sys
import numpy as np
import read_dscovr as rd

def lcdscovr(mode,lcindex=0,Nreduce=4,Nfin=None,datadir="/home/kawahara/exomap/data/for_HKawahara"):
    #Pick LC in every Nreduce 
    W,obst,lcall,timeobj=rd.read_dscovr(datadir,Nreduce,Nfin,istart=3,timeobj=True)
    
    if mode=="unmix":
        import read_data
        lcall=lcall*npix
        lcall=lcall/6 #(Ni, Nl)
        Ni,Nl=np.shape(lcall)
        theme="dscovr"
        axfile="/home/kawahara/exomap/gpmap/dscovr/D203L2-VRDet_A-2.0X-4.5j40000.npz"
        A,X,resall=read_data.readax(axfile)
        M=uc.unmix_curve(X,lcall)
        lc=M[:,lcindex]
    elif mode=="pca":
        from sklearn.decomposition import PCA
        Ni,Nl=np.shape(lcall)
        pca = PCA(n_components=Nl, svd_solver='full')
        pca.fit(lcall)
        Xp=((pca.components_))
#        for l in range(0,Nl):
#            plt.plot(Xp[l,:])
        lc=lcall@Xp[lcindex,:]        
    elif mode=="normal":        
        Ni,Nl=np.shape(lcall)
        lc=lcall[:,lcindex]
    else:
        print("No mode")
        sys.exit()
        
    return W, obst, lc, timeobj

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    W, obst, lc, timeobj=lcdscovr("pca",Nreduce=2,Nfin=2435)
    #4870
    print(len(lc))
    plt.plot(lc)
    plt.savefig("pc1.png")
    plt.show()
