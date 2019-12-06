import numpy as np
import colour 
import healpy as hp

def convimg2rgb(ti):
    XYZ = colour.sRGB_to_XYZ(ti)
    xy = colour.XYZ_to_xy(XYZ)
    return xy,ti


def generate_palette(nside=16):
    rgb=[]
    #    for ipix in range(0,hp.nside2npix(nside)):
    npix=hp.nside2npix(nside)
    theta,phi=hp.pix2ang(nside,range(0,npix))
    rgb=np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)

    rgb=np.array(rgb).T
    mask=(np.min(rgb,axis=1)>0.0)
    return rgb[mask,:]
    

def color_weight_function(w,nside=16):
    rgb=generate_palette(nside)
    Mk,Nl=np.shape(rgb)
    
    return

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rgb=generate_palette(16)
    print(np.shape(rgb))
    
    fig=plt.figure()
    ax=fig.add_subplot(121)
    ax.plot(rgb[:,0],rgb[:,1],".")
    
    XYZ = colour.sRGB_to_XYZ(rgb)
    xy = colour.XYZ_to_xy(XYZ)    
    ax=fig.add_subplot(122,aspect=1.0)
    

    ax.scatter(xy[:,0], xy[:,1],facecolor=rgb,alpha=1,s=2)
    ax.set_xlim(0.15,0.62)
    ax.set_ylim(0.2,0.62)
    ax.set_title("reconstruct")
    plt.show()
    
