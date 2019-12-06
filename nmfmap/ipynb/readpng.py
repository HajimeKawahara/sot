#!/usr/bin/python
import pylab
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import module
import colorvector

def toycar(Nreduce=16,Npad=8):
    img=get_img("./car.png")
    timg=img[::Nreduce,::Nreduce,:]
    Nx, Ny, Nrgb = np.shape(timg)
    N_data = 100
    rand_now = module.random_generator(N_data, Nx, Ny)
    dRGB, g=rand_now.make_colordata(timg,20) 
    g=np.array(g)
    rgbvec=colorvector.generate_palette(Npad)
    Npal=np.shape(rgbvec)[0]
    I_init=np.ones((Nx,Ny,Npal))
    gall=np.einsum("ijk,cl->ijkcl",g,rgbvec)
    return timg,rgbvec,I_init,gall,dRGB

    
def get_bwimg(file):
    bwimg=mpimg.imread(file)
    bwimg=np.sum(bwimg,axis=2)/3.0    
    return bwimg


def get_img(file):
    img=mpimg.imread(file)
    #bwimg=np.sum(bwimg,axis=2)/3.0    
    return img[:,:,0:3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', nargs=1, required=True, help='png file')
    args = parser.parse_args()    
    img=get_bwimg(args.f[0])
    print (img.shape)
    fig =plt.figure()
    imshow(img,cmap="gray")
    plt.show()
