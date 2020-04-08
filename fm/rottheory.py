#!/usr/bin/python
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pylab 
import numpy as np
import healpy as hp
import time
import read_binaryearth as rb
import scipy.signal

def modmax(Theta,zeta,inc,Thetaeq):
    fac1=np.cos(zeta) + np.cos(zeta)*np.cos(Theta)*np.sin(inc)-np.cos(inc)*np.sin(zeta)*np.sin(Theta - Thetaeq)
    fac2=np.cos(Theta-Thetaeq)**2 + 2*np.cos(Theta-Thetaeq)*np.cos(Thetaeq)*np.sin(inc)+np.cos(Thetaeq)**2*np.sin(inc)**2+(np.cos(inc)*np.sin(zeta) - np.cos(zeta)*np.sin(Theta-Thetaeq) + np.cos(zeta)*np.sin(inc)*np.sin(Thetaeq))**2
    return fac1/fac2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate LC')
    args = parser.parse_args()    
    
    print modmax(np.pi/2.0,np.pi/2.0,np.pi/4.0,np.pi)
