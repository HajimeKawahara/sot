"""generates a moving map

"""
from sot.core import mapops 
import numpy as np
from tqdm import tqdm

def rotating_map(mmap,obst,rotphimax=0.0,rotthetamax=3*np.pi/4):
    """generates a rotating map
    
    Args:
        mmap (array): map (Nj, ) or (Nj, Nl)
        obst (array): time series (Ni,)
        rotphimax (float, optional): _description_. Defaults to 0.0.
        rotthetamax (_type_, optional): _description_. Defaults to 3*np.pi/4.

    Returns:
        array: moving map (Ni, Nj) or (Nj, Ni, Nl)
    """
    Ni=len(obst)
    rotphi=np.linspace(0.0,rotphimax,Ni)
    rottheta=np.linspace(0.0,rotthetamax,Ni)
    ndim=np.shape(np.shape(mmap))[0]
    if ndim==1:
        movmap=[]
        for i in range(0,Ni):
            movmap.append(mapops.rotate_map(mmap, rottheta[i], rotphi[i]))
        movmap=np.array(movmap)
    elif ndim==2:
        _,Nl=np.shape(mmap)

        movmap=[]
        for i in tqdm(range(0,Ni)):
            MM=[]
            for l in (range(0,Nl)):
                MM.append(mapops.rotate_map(mmap[:,l], rottheta[i], rotphi[i]))
            movmap.append(np.array(MM).T)
        movmap=np.array(movmap)
        
    return movmap

def sinmap(mmap,obst,Ns=10):
    err=0.3
    Nj=len(mmap)
    m=[]
    sa=np.random.rand(Ns)
    fa=np.random.rand(Ns)*0.5
    trend=samplefunc(obst/4,sa,fa,err)/10
    for j in range(0,Nj):
        if mmap[j]>0.0:
            mtmp=mmap[j]+trend
        else:
            mtmp=mmap[j]+trend*0.0
        m.append(mtmp)
    M=np.array(m).T
    return M

def samplefunc(tarr,sa,fa,err):
    y=[]
    for t in tarr:
        y.append(np.sum(sa*np.sin(2*np.pi*fa*t),axis=0))
    return np.array(y)-np.mean(y)+np.random.normal(0,err,len(y))

