import numpy as np
import scipy
import sys

def RBF(obst,tau):
    if np.shape(np.shape(obst))[0]==1:
        Dt = obst - np.array([obst]).T
    elif np.shape(np.shape(obst))[0]==2:
        Dt = obst
    K=np.exp(-(Dt)**2/2/(tau**2))
    return K

def Matern32(obst,tau):
    if np.shape(np.shape(obst))[0]==1:
        Dt = obst - np.array([obst]).T
    elif np.shape(np.shape(obst))[0]==2:
        Dt = obst
    fac=np.sqrt(3.0)*np.abs(Dt)/tau
    K=(1.0+fac)*np.exp(-fac)
    return K

def minuslogRBF(obst,tau):
    if np.shape(np.shape(obst))[0]==1:
        Dt = obst - np.array([obst]).T
    elif np.shape(np.shape(obst))[0]==2:
        Dt = obst
    K=(Dt)**2/2/(tau**2)
    return K


def d_Matern32(obst,tau):
    #dK/d acute_tau, tau=exp(tau_acute), tau_acute=log(tau)
    if np.shape(np.shape(obst))[0]==1:
        Dt = obst - np.array([obst]).T
    elif np.shape(np.shape(obst))[0]==2:
        Dt = obst
    fac=-np.sqrt(3.0)*np.abs(Dt)/tau-2.0*np.log(tau)
    dK=3.0*np.abs(Dt)**2*np.exp(fac)
    return dK

def d_RBF(obst,tau):
    #tau_acute=log(tau), tau=exp(tau_acute)
    if np.shape(np.shape(obst))[0]==1:
        Dt = obst - np.array([obst]).T
    elif np.shape(np.shape(obst))[0]==2:
        Dt = obst
    Dt2=(Dt)**2
    fac=-2.0*np.log(tau)-Dt2/(tau**2)/2.0
    dK=Dt2*np.exp(fac)
    
    return dK

#### L2 wrapping to GP kernel #####
def Wrap_Ridge(K,alpha,lam):
    Nt=np.shape(K)[0]
    Kc=lam**2*((1.0-alpha)*K+alpha*np.eye(Nt))
    return Kc

def d_Wrap_Matern32(alpha,lam,tau,obst):
    Nt=len(obst)
    lam2=lam*lam
    dtau_acute=d_Matern32(obst,tau)*lam2*(1.0-alpha)
    K=Matern32(obst,tau)
    #alpha=expit(alpha_acute)=1/(1+exp(-alpha_acute))
    alpha_acute=scipy.special.logit(alpha)
    dalpha_acute=lam2*(np.eye(Nt)-K)*(alpha**2)*np.exp(-alpha_acute)
    dlam2=(1.0-alpha)*K+alpha*np.eye(Nt)
    dkernel=np.array([dalpha_acute,dlam2,dtau_acute])
    return dkernel

