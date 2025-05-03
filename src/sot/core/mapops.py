"""map operators"""

import healpy as hp
import numpy as np

def rotate_map(hmap, rot_theta, rot_phi):
    """rotate a healpy map
    Args:
        hmap (array): healpy map
        rot_theta (float): rotation angle in theta direction
        rot_phi (float): rotation angle in phi direction
    Returns:
        array: rotated healpy map
    """
    nside = hp.npix2nside(len(hmap))
    t,p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    r = hp.Rotator(deg=False, rot=[rot_phi,rot_theta])
    trot, prot = r(t,p)
    rmap = hp.get_interp_val(hmap, trot, prot)
    return rmap
