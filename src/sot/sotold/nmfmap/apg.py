import cupy as cp
import numpy as np

#
# APGr: Accelerated Projected Gradient + restart
#
# Lipfac=reducing factor of the descent step.
# If the result drops zero-vectors, set smalle Lipfac.  
#
def APGr(n,Q,p,x0,Ntry=1000,alpha0=0.9,eta=0.0, Lip="frobenius",Lipfac=1.0):
    if Lip=="frobenius":
        normQ = cp.sqrt(cp.sum(Q**2))/Lipfac
    elif Lip=="norm2":        
        normQ = np.linalg.norm(cp.asnumpy(Q),2)/Lipfac
#    print("descent gradient=",1.0/normQ)
    Theta1 = cp.eye(n) - Q/normQ
    theta2 = p/normQ
    x = cp.copy(x0)
    y = cp.copy(x0)
    x[x<0]=0.0
    
    alpha=alpha0
    cost0=0.5*cp.dot(x0,cp.dot(Q,x0)) - cp.dot(p,x0)
    costp=cost0
    for i in range(0,Ntry):
        xp=cp.copy(x)
        x = cp.dot(Theta1,y) + theta2
        x[x<0] = 0.0
        
        dx=x-xp
        aa=alpha*alpha
        beta=alpha*(1.0-alpha)
        alpha=0.5*(np.sqrt(aa*aa + 4*aa) - aa)
        beta=beta/(alpha + aa)
        y=x+beta*dx
        cost=0.5*cp.dot(x,cp.dot(Q,x)) - cp.dot(p,x)
        if cost > costp:
            x = cp.dot(Theta1,xp) + theta2
            x[x<0] = 0.0            
            y = cp.copy(x)
            alpha=alpha0
        elif costp - cost < eta:            
            #print(i,cost0 - cost)
            return x

        costp=cp.copy(cost)
        if cost != cost:
            print("Halt at APGr")
            print("Q,p,x0",Q,p,x0)
            print("cost=",cost)
            sys.exit()
            
    #print(i,cost0 - cost)
    return x

#
#AGr: Accelerated (non-projected) Gradient + restart for semi NMF
#Lipfac
#
def AGr(n,Q,p,x0,Ntry=1000,alpha0=0.9,eta=0.0, Lip="frobenius",Lipfac=1.0):
    if Lip=="frobenius":
        normQ = cp.sqrt(cp.sum(Q**2))*Lipfac
    elif Lip=="norm2":        
        normQ = np.linalg.norm(cp.asnumpy(Q),2)*Lipfac
    Theta1 = cp.eye(n) - Q/normQ
    theta2 = p/normQ
    x = cp.copy(x0)
    y = cp.copy(x0)
    x[x<0]=0.0
    alpha=alpha0
    cost0=0.5*cp.dot(x0,cp.dot(Q,x0)) - cp.dot(p,x0)
    costp=cost0
    for i in range(0,Ntry):
        xp=cp.copy(x)
        x = cp.dot(Theta1,y) + theta2
        x[x<0] = 0.0
        dx=x-xp
        aa=alpha*alpha
        beta=alpha*(1.0-alpha)
        alpha=0.5*(np.sqrt(aa*aa + 4*aa) - aa)
        beta=beta/(alpha + aa)
        y=x+beta*dx
        cost=0.5*cp.dot(x,cp.dot(Q,x)) - cp.dot(p,x)
        if cost > costp:
            x = cp.dot(Theta1,xp) + theta2
            y = cp.copy(x)
            alpha=alpha0
        elif costp - cost < eta:
            print(i,cost0 - cost)

            return x

        costp=cp.copy(cost)
        if cost != cost:
            print("Halt at AG")
            print("Q,p,x0",Q,p,x0)
            print("cost=",cost)
            sys.exit()
            
    print(i,cost0 - cost)
#    edd=time.time()
#    print("APG",edd-ed,"sec")

    return x
