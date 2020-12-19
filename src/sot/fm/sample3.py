import sys
import argparse
import numpy as np
import rottheory as rt


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

    
    arr=rt.modmax(gridT,gridz,args.i[0]/180*np.pi,args.t[0]/180*np.pi)
    print(np.shape(arr))
    fig=plt.figure()
    plt.contour(X,Y,arr,colors="white",levels=[-1.25,-1.0,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.0,1.25])
    c=plt.imshow(arr,cmap="twilight_shifted",vmin=-1.0,vmax=1.0,extent=[0,2*np.pi,np.pi,0])
    plt.colorbar(c)
    plt.gca().invert_yaxis()
    plt.ylabel("$\zeta$",fontsize=15)
    plt.xlabel("$\Theta$",fontsize=15)
    plt.tick_params(labelsize=15)
    plt.savefig("rott_i"+str(args.i[0])+"_t"+str(args.t[0])+".png")
    plt.show()
