import numpy as np
import matplotlib.pyplot as plt
import read_data

if __name__=='__main__':
    #
    #    Comparing prediction and light curve
    #
    import sys
    axfile=sys.argv[1]
    #axfile=npz/T215/T215_N3_L2-VRDet_A-1.0X2.0j100000.npz
    lcall=np.load("lcallN0.01.npz")["arr_0"]
    A,X,resall=read_data.readax(axfile)
    W=np.load("w512.npz")["arr_0"]

    WAX=W@A@X
    res=lcall-WAX
    fig=plt.figure()
    ax=fig.add_subplot(211)
    plt.plot(lcall[:,0])
    plt.plot(WAX[:,0],".")
    ax=fig.add_subplot(212)
    plt.plot(res[:,0])
    plt.show()

