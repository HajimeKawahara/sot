import healpy as hp
import numpy as np

def rotate_map(hmap, rot_theta, rot_phi):
    nside = hp.npix2nside(len(hmap))
    t,p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    r = hp.Rotator(deg=False, rot=[rot_phi,rot_theta])
    trot, prot = r(t,p)
    rmap = hp.get_interp_val(hmap, trot, prot)
    return rmap
