#!/usr/bin/python
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyhdf.SD import SD,SDC
import healpy as hp
import os 
def modisfile(month,day=None,year=2016,datadir="/home/kawahara/exomap/sot/data/aux/"):
    import pandas as pd
    if day is None:
        dat=pd.read_csv(os.path.join(datadir,"modisM.txt"),delimiter=",")
        mask=dat["YEAR"]==2016
        mask2=dat["MONTH"]==month
        ext=(dat[mask&mask2])
        hdffile=(ext["FILE"])
        out=str(hdffile.tolist()[0])
        day=None
    else:
        dat=pd.read_csv(os.path.join(datadir,"modisE.txt"),delimiter=",")
        try:
            mask=dat["YEAR"]==2016
            mask2=dat["MONTH"]==month
            mask3=dat["DAY"]==day
            ext=(dat[mask&mask2&mask3])
            hdffile=(ext["FILE"])
            out=str(hdffile.tolist()[0])
        except:
            mask=dat["YEAR"]==2016
            mask2=dat["MONTH"]==month
            ext=(dat[mask&mask2])
            i=np.argmin(np.abs(ext["DAY"]-day))
            hdffile=(ext["FILE"])
            out=str(hdffile.tolist()[i])
            day=ext["DAY"].tolist()[i]
            print("Nearest Day is used day=",day)
    return out,month,day

def read_cloud(hdffile,N=1):
    print(hdffile)
    f = SD(hdffile,SDC.READ)
    v=f.select("Cloud_Fraction_Mean_Mean")
    vr=v.attributes()["valid_range"]
    fv=v.attributes()["_FillValue"]
    ao=v.attributes()["add_offset"]
    sf=v.attributes()["scale_factor"]
    
    a=np.array(v[::N,::N],dtype=float)
    a[a==fv] = None
    a=(a-ao)*sf
    return a

def to_healpix(a,nside=16):
    Nphi,Ntheta=np.shape(a)
    npix=hp.nside2npix(nside)
    hmap=np.zeros(npix)
    for ipix in range(0,npix):
        theta,phi=hp.pix2ang(nside,ipix)        
        itheta=int(theta/np.pi*180)
        iphi=int(phi/np.pi*180)
        hmap[ipix]=float(a[itheta,iphi])
    return hmap

if __name__ == "__main__":
    import os
    import plotdymap
    import matplotlib
    fontsize=16
    matplotlib.rcParams.update({'font.size':fontsize})

    thetaE,phiE=plotdymap.bound_earth("/home/kawahara/exomap/sot/data/earth_boundary.npz")
    year=2016
    month=5
    day=11
    hdffile,month,day=modisfile(month,day,year=year)
    hdfdir="/home/kawahara/exomap/data/modis/MYD08"

    hdffile=os.path.join(hdfdir,hdffile)
    a=read_cloud(hdffile,N=1)
    hmap=to_healpix(a,nside=16)
    hp.mollview(hmap, title=str(year)+"-"+str(month)+"-"+str(day),flip="geo",cmap=plt.cm.pink,min=0.5,max=1.0)
    hp.projplot(thetaE, phiE,".",c="#66CC99")
    hp.projtext(-60,-25,"A",coord="G",lonlat=True,color="cyan",fontsize=26) #amazon
#    hp.projtext(-130,30,"B",coord="G",lonlat=True,color="cyan",fontsize=26) #north america

    plt.savefig("cf"+str(year)+"_"+str(month)+"_"+str(day)+".pdf")
    plt.show()
