import numpy as np


def mrsa(vec1,vec2):
    v1=(vec1-np.mean(vec1))
    v2=(vec2-np.mean(vec2))
    v1norm=np.sqrt(np.dot(v1,v1))
    v2norm=np.sqrt(np.dot(v2,v2))
    naib=np.dot(v1,v2)/v1norm/v2norm
    if naib>1.0:
        naib=1.0
    return 1.0/np.pi*np.arccos(naib)

def mrsa_mean(X):
    Xini=np.load("Xinit.npz")["arr_0"]
    mrsa_arr=[]
    for i in range(0,3):
        ma=[]
        for j in range(0,3):
            ma.append(mrsa(Xini[i,:],X[j,:]))
        minma=np.min(np.array(ma))
        mrsa_arr.append(minma)
    mrsa_arr=np.array(mrsa_arr)
    return (np.mean(mrsa_arr))

if __name__=='__main__':
    import read_data
    import sys
    #    axfile="npz/T116/T116_L2-VRLD_A-2.0X4.0j99000.npz"
    axfile=sys.argv[1]
    A,X,resall=read_data.readax(axfile)
    mmrsa=mrsa_mean(X)
    print(mmrsa)
