import healpy as hp
import numpy as np
import time 

def calc_sepmatrix(nside):
    v = hp.pix2vec(nside, np.arange(0, hp.nside2npix(nside)))
    v = np.column_stack(v)
    cossep = np.sum(v[np.newaxis,:,:]*v[:,np.newaxis,:], axis=2)
    cossep[cossep > 1] = 1.0
    cossep[cossep < -1] = -1.0    
    thetas = np.arccos(cossep)
    return thetas

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sepmat=calc_sepmatrix(16)
