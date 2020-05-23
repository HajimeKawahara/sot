#Generating a moving map

import rotmap
import numpy as np
from tqdm import tqdm

def rotating_map(mmap,obst,rotphimax=0.0,rotthetamax=3*np.pi/4):
    Nt=len(obst)
    rotphi=np.linspace(0.0,rotphimax,Nt)
    rottheta=np.linspace(0.0,rotthetamax,Nt)
    ndim=np.shape(np.shape(mmap))[0]
    if ndim==1:
        M=[]
        for i in range(0,Nt):
            M.append(rotmap.rotate_map(mmap, rottheta[i], rotphi[i]))
        M=np.array(M)
    elif ndim==2:
        Nj,Nl=np.shape(mmap)

        M=[]
        for i in tqdm(range(0,Nt)):
            MM=[]
            for l in (range(0,Nl)):
                MM.append(rotmap.rotate_map(mmap[:,l], rottheta[i], rotphi[i]))
            M.append(np.array(MM).T)
        M=np.array(M)
        
    return M

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

