import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
def classdif(Aclass):
    dataclass=np.load("/home/kawahara/exomap/sot/data/cmap3class.npz")
    cmapans=dataclass["arr_0"]
    #    hp.mollview(cmapans, title="Input",flip="geo",cmap=plt.cm.Greys_r)
    Ax=hp.ud_grade(Aclass,32)
    #    hp.mollview(Ax, title="Input",flip="geo",cmap=plt.cm.Greys_r)
    #    plt.show()
    npix=hp.nside2npix(32)
    dif=np.abs(Ax-cmapans)
    matchrate=(len(dif[dif==0.0])/npix)
    return matchrate

def mclassmap(A):
    Aclass=np.argmax(A,axis=1)
    Aclass[Aclass==0]=30
    Aclass[Aclass==1]=100
    Aclass[Aclass==2]=0
    return Aclass
