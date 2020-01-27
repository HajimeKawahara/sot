
import numpy as np
def get_axfiles_X():
    axfiles=[\
             "npz/T120E/T120_L2-VRDet_A-2.0X-0.5Ej1092.npz",\
             "npz/T120E/T120_L2-VRDet_A-2.0X0.0Ej1085.npz",\
             "npz/T120E/T120_L2-VRDet_A-2.0X0.5Ej1179.npz",\
             "npz/T120E/T120_L2-VRDet_A-2.0X1.0Ej137048.npz",\
             "npz/T120E/T120_L2-VRDet_A-2.0X1.5Ej78480.npz",\
             "npz/T120E/T120_L2-VRDet_A-2.0X2.0Ej24524.npz"\
    ]
    lamx=[-0.5,0,0.5,1,1.5,2.0]
    lamx=np.array(lamx)
    return axfiles,lamx

def get_axfiles_A():
    axfiles=[\
#             "npz/T120E/T120_L2-VRDet_A0.0X1.0Ej310.npz",\
             "npz/T120E/T120_L2-VRDet_A-0.5X1.0Ej334.npz",\
             "npz/T120E/T120_L2-VRDet_A-1.0X1.0Ej517.npz",\
             "npz/T120E/T120_L2-VRDet_A-1.5X1.0Ej921.npz",\
             "npz/T120E/T120_L2-VRDet_A-2.0X1.0Ej137048.npz",\
             "npz/T120E/T120_L2-VRDet_A-2.5X1.0Ej260886.npz",\
             "npz/T120E/T120_L2-VRDet_A-3.0X1.0Ej264283.npz",\
             "npz/T120E/T120_L2-VRDet_A-3.5X1.0Ej261453.npz",\
             "npz/T120E/T120_L2-VRDet_A-4.0X1.0Ej268552.npz"             
    ]
    lama=[-0.5,-1,-1.5,-2,-2.5,-3,-3.5,-4]
    lama=np.array(lama)
    return axfiles,lama