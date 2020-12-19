import sys
import argparse
import numpy as np

def modmax(Theta,zeta,inc,Thetaeq):
    fac1=- np.cos(zeta) + np.cos(inc)*np.sin(zeta)*np.sin(Theta - Thetaeq) - np.cos(zeta)*np.cos(Theta)*np.sin(inc)
    fac2=np.cos(Theta-Thetaeq)**2 + 2*np.cos(Theta-Thetaeq)*np.cos(Thetaeq)*np.sin(inc)+np.cos(Thetaeq)**2*np.sin(inc)**2+(np.cos(inc)*np.sin(zeta) - np.cos(zeta)*np.sin(Theta-Thetaeq) + np.cos(zeta)*np.sin(inc)*np.sin(Thetaeq))**2
    return fac1/fac2

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import pylab 
    
    parser = argparse.ArgumentParser(description='max approx')
    parser.add_argument('-i', nargs=1, default=[0.0], help='inclination [deg]', type=float)
    parser.add_argument('-t', nargs=1, default=[0.0], help='Thetaeq [deg]', type=float)
    args = parser.parse_args()    

    N=180
    gTheta=np.linspace(0,2*np.pi,2*N)
    gzeta=np.linspace(0,np.pi,N)
    X,Y=np.meshgrid(gTheta,gzeta)
    gridT=(gTheta*np.array([np.ones(N)]).T)
    gridz=(gzeta*np.array([np.ones(2*N)]).T).T

    
    arr=modmax(gridT,gridz,args.i[0]/180*np.pi,args.t[0]/180*np.pi)
    print(np.shape(arr))
    fig=plt.figure()
    plt.contour(X,Y,arr,colors="white",levels=[-1.25,-1.0,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.0,1.25])
    plt.imshow(arr,cmap="coolwarm",vmin=-1.0,vmax=1.0,extent=[0,2*np.pi,np.pi,0])
    plt.gca().invert_yaxis()
    plt.ylabel("$\zeta$",fontsize=15)
    plt.xlabel("$\Theta$",fontsize=15)
    plt.tick_params(labelsize=15)
    plt.savefig("rott.png")
    plt.show()
