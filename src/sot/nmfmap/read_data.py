import numpy as np
#import cupy as cp
def readax(axfile):
    dat=np.load(axfile)
    A=dat["arr_0"]
    X=dat["arr_1"]
    try:
        resall=dat["arr_2"]
    except:
        resall=[]
    return A,X,resall

def getband():
    bands=[[0.4,0.45],[0.45,0.5],[0.5,0.55],[0.55,0.6],[0.6,0.65],[0.65,0.7],[0.7,0.75],[0.75,0.8],[0.8,0.85],[0.85,0.9]]
    return bands
