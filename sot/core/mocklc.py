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

def comp_omega(nside):
    omega = []
    npix = hp.nside2npix(nside)
    #print(("npix=", npix))
    for ipix in range(0, npix):
        theta, phi = hp.pix2ang(nside, ipix)
        omega.append([theta, phi])
    return np.array(omega)


def uniteO(inc, Thetaeq):
    # (3)
    eO = np.array([np.sin(inc)*np.cos(Thetaeq), -
                   np.sin(inc)*np.sin(Thetaeq), np.cos(inc)])
    return eO


def uniteS(Thetaeq, Thetav):
    # (3,nsamp)
    eS = np.array(
        [np.cos(Thetav-Thetaeq), np.sin(Thetav-Thetaeq), np.zeros(len(Thetav))])
    return eS


def uniteR(zeta, Phiv, omega):
    # (3,nsamp,npix)
    np.array([Phiv]).T
    costheta = np.cos(omega[:, 0])
    sintheta = np.sin(omega[:, 0])
    cosphiPhi = np.cos(omega[:, 1]+np.array([Phiv]).T)
    sinphiPhi = np.sin(omega[:, 1]+np.array([Phiv]).T)
#    cosphiPhi=np.cos(omega[:,1]-np.array([Phiv]).T)
#    sinphiPhi=np.sin(omega[:,1]-np.array([Phiv]).T)

    x = cosphiPhi*sintheta
    y = np.cos(zeta)*sinphiPhi*sintheta+np.sin(zeta)*costheta
    z = -np.sin(zeta)*sinphiPhi*sintheta+np.cos(zeta)*costheta
    eR = np.array([x, y, z])

    return eR


def comp_weight(nside, zeta, inc, Thetaeq, Thetav, Phiv):
    omega = comp_omega(nside)

    eO = uniteO(inc, Thetaeq)
    eS = uniteS(Thetaeq, Thetav)
    eR = uniteR(zeta, Phiv, omega)

    #print("Shapes of eO, eS, eR")
    #print(np.shape(eO), np.shape(eS), np.shape(eR))

#    start = time.time()
    WV = []
    for ir in (range(0, np.shape(eS)[1])):
        ele = np.dot(eR[:, ir, :].T, eO)
        WV.append(ele)
    WV = np.array(WV)
#    WV=np.dot(eR.T,eO)
    mask = (WV < 0.0)
    WV[mask] = 0.0

#    elapsed_time = time.time() - start
#    print ("elapsed_time (eR.eO) :{0}".format(elapsed_time)) + "[sec]"
#    print "Shape of WV = ",np.shape(WV)

#    start = time.time()
#    print np.shape(eR), np.shape(eS)
    WI = []
    for ir in (range(0, np.shape(eS)[1])):
        ele = np.dot(eR[:, ir, :].T, eS[:, ir])
        WI.append(ele)
    WI = np.array(WI)
 #   print np.shape(WI)
    mask = (WI < 0.0)
    WI[mask] = 0.0
#    elapsed_time = time.time() - start
#    print ("elapsed_time (eR.eS) :{0}".format(elapsed_time)) + "[sec]"
#    print "Shape of WI = ",np.shape(WI)

    return WI, WV


def comp_weight_light(x, worb, wspin):
    Thetav = worb*x
    Phiv = np.mod(wspin*x, 2*np.pi)
