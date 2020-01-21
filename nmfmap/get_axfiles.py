import numpy as np
def get_axfiles_X():
    axfiles=[\
             "npz/T120/T120_L2-VRDet_A-2.0X0.0Ej1085.npz",\
             "npz/T120/T120_L2-VRDet_A-2.0X0.5Ej1179.npz",\
             "npz/T120/T120_L2-VRDet_A-2.0X1.0Ej137048.npz",\
             "npz/T120/T120_L2-VRDet_A-2.0X1.5Ej78480.npz",\
             "npz/T120/T120_L2-VRDet_A-2.0X2.0Ej24524.npz"\
    ]
    lamx=[0,0.5,1,1.5,2.0]
    lamx=np.array(lamx)
    return axfiles,lamx

def get_axfiles_A():
    axfiles=[\
             "npz/T120/T120_L2-VRDet_A-1.0X1.0Ej517.npz",\
             "npz/T120/T120_L2-VRDet_A-1.5X1.0Ej921.npz",\
             "npz/T120/T120_L2-VRDet_A-2.0X1.0Ej137048.npz",\
             "npz/T120/T120_L2-VRDet_A-3.0X1.0Ej264283.npz"\
    ]
    lama=[-1,-1.5,-2,-3]
    lama=np.array(lama)
    return axfiles,lama
