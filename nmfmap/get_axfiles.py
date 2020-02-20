
import numpy as np
def get_axfiles_X():
    axfiles=[\
             "npz/T215/T215_N3_L2-VRDet_A-1.0X-2.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A-1.0X-1.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A-1.0X0.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A-1.0X1.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A-1.0X2.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A-1.0X3.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A-1.0X4.0j100000.npz"
    ]
    lamx=[-2.0,-1.0,0.0,1,2,3,4]
    lamx=np.array(lamx)
    return axfiles,lamx

def get_axfiles_A():
    axfiles=[\
             "npz/T215/T215_N3_L2-VRDet_A-4.0X2.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A-3.0X2.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A-2.0X2.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A-1.0X2.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A0.0X2.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A1.0X2.0j100000.npz",\
             "npz/T215/T215_N3_L2-VRDet_A2.0X2.0j100000.npz"
    ]
    lama=[-4,-3,-2,-1,0.0,1.0,2.0]
    lama=np.array(lama)
    return axfiles,lama


def get_axfiles_Ad():
    axfiles=[\
             "npz/D203/D203L2-VRDet_A-2.0X-3.0j40000.npz",\
             "npz/D203/D203L2-VRDet_A-2.0X-4.0j40000.npz",\
             "npz/D203/D203L2-VRDet_A-2.0X-4.5j40000.npz",\
             "npz/D203/D203L2-VRDet_A-2.0X-5.0j40000.npz",\
             "npz/D203/D203L2-VRDet_A-2.0X-6.0j40000.npz",\
             "npz/D203/D203L2-VRDet_A-2.0X-7.0j40000.npz",\
             "npz/D203/D203L2-VRDet_A-2.0X-8.0j40000.npz"
    ]
    lama=[-3.0,-4.0,-4.5,-5.0,-6.0,-7.0,-8.0]
    lama=np.array(lama)
    return axfiles,lama

def get_axfiles_Xd():
    axfiles=[\
             "npz/D203n/D203L2-VRDet_A-1.0X-4.5j40000.npz",\
             "npz/D203n/D203L2-VRDet_A-1.5X-4.5j40000.npz",\
             "npz/D203/D203L2-VRDet_A-2.0X-4.5j40000.npz",\
             "npz/D203n/D203L2-VRDet_A-2.5X-4.5j40000.npz",\
             "npz/D203n/D203L2-VRDet_A-3.0X-4.5j40000.npz"
    ]
    lamx=[-1.0,-1.5,-2.0,-2.5,-3.0]
    lamx=np.array(lamx)
    return axfiles,lamx
