#!/usr/bin/python
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import healpy as hp
import time
import scipy.signal
from scipy.interpolate import splev, splrep
import tqdm
import jax.numpy as jnp
from jax import jit

def comp_omega(nside):
    omega = []
    npix = hp.nside2npix(nside)
    #print(("npix=", npix))
    for ipix in range(0, npix):
        theta, phi = hp.pix2ang(nside, ipix)
        omega.append([theta, phi])
    return jnp.array(omega)

@jit
def uniteO(inc, Thetaeq):
    # (3)
    eO = jnp.array([jnp.sin(inc)*jnp.cos(Thetaeq), -
                   jnp.sin(inc)*jnp.sin(Thetaeq), jnp.cos(inc)])
    return eO

@jit
def uniteS(Thetaeq, Thetav):
    # (3,nsamp)
    eS = jnp.array(
        [jnp.cos(Thetav-Thetaeq), jnp.sin(Thetav-Thetaeq), jnp.zeros(len(Thetav))])
    return eS

@jit
def uniteR(zeta, Phiv, omega):
    # (3,nsamp,npix)
    jnp.array([Phiv]).T
    costheta = jnp.cos(omega[:, 0])
    sintheta = jnp.sin(omega[:, 0])
    cosphiPhi = jnp.cos(omega[:, 1]+jnp.array([Phiv]).T)
    sinphiPhi = jnp.sin(omega[:, 1]+jnp.array([Phiv]).T)
#    cosphiPhi=np.cos(omega[:,1]-np.array([Phiv]).T)
#    sinphiPhi=np.sin(omega[:,1]-np.array([Phiv]).T)

    x = cosphiPhi*sintheta
    y = jnp.cos(zeta)*sinphiPhi*sintheta+jnp.sin(zeta)*costheta
    z = -jnp.sin(zeta)*sinphiPhi*sintheta+jnp.cos(zeta)*costheta
    eR = jnp.array([x, y, z])

    return eR

@jit
def comp_weight(nside,zeta,inc,Thetaeq,Thetav,Phiv):
    omega=comp_omega(nside)
    eO=uniteO(inc,Thetaeq)
    eS=uniteS(Thetaeq,Thetav)
    eR=uniteR(zeta,Phiv,omega)
    WV=jnp.einsum("ijk,i->jk",eR,eO)
    WV=jnp.where(WV<0.0,0.0,WV)
    WI=jnp.einsum("ijk,ij->jk",eR,eS)
    WI=jnp.where(WI<0.0,0.0,WI)
    return WI,WV

