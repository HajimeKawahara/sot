import numpy as np
import healpy as hp
import io_refdata


def read_classification(filename):
    data = np.loadtxt(filename, skiprows=6)
    ydim = 1776
    xdim = 4320
    cmap = data.reshape(ydim, xdim)
    return cmap


def merge_to_4classes(cmap, val_desert=100, val_snow=90, val_veg=30):
    ydim = 1776
    xdim = 4320
    val_ocean = 0
    # 4 categories classification
    # desert
    mask = (cmap > 15)+(cmap == 7)
    cmap[mask] = val_desert
    # vegetation
    mask = (cmap > 0)*(cmap < 15)
    cmap[mask] = val_veg
    # snow/ice
    mask = (cmap == 15)
    cmap[mask] = val_snow
    vals = np.array([val_ocean, val_desert, val_snow, val_veg])
    valexp = ["ocean", "desert", "snow", "veg"]
    return cmap, vals, valexp

def merge_to_3classes(cmap, val_desert=100, val_veg=30):
    ydim = 1776
    xdim = 4320
    val_ocean = 0
    # 4 categories classification
    # desert
    mask = (cmap > 15)+(cmap == 7)
    cmap[mask] = val_desert
    # vegetation
    mask = (cmap > 0)*(cmap < 15)
    cmap[mask] = val_veg
    # snow/ice
    mask = (cmap == 15)
    cmap[mask] = val_ocean
    vals = np.array([val_ocean, val_desert, val_veg])
    valexp = ["ocean", "desert", "veg"]
    return cmap, vals, valexp


def set_meanalbedo(waves, wavee, refsurfaces, sky, onsky=False):
    ma = []
    if onsky:
        atm = io_refdata.get_meanalbedo(sky, waves, wavee)
        for i in range(0, len(refsurfaces)):
            ma.append(io_refdata.get_meanalbedo(refsurfaces[i], waves, wavee)+atm)
    else:
        for i in range(0, len(refsurfaces)):
            ma.append(io_refdata.get_meanalbedo(refsurfaces[i], waves, wavee))
        
    return np.array(ma)


def copy_to_healpix(cmap, nside=16):
    npix = hp.nside2npix(nside)
    arr = np.zeros(npix)
    ydim, xdim = np.shape(cmap)
    for ipix in range(0, npix):
        theta, phi = hp.pix2ang(nside, ipix)  # theta:0-pi, phi:0-2pi
        itheta = int(theta/np.pi*ydim)
        iphi = int(np.mod(phi+np.pi,2*np.pi)/(2*np.pi)*xdim)
        arr[ipix] = cmap[itheta, iphi]
    return arr, nside

def plot_albedo(veg,soil,cloud,snow_med,water,clear_sky,ave_band,malbedo,valexp):
    import matplotlib.pyplot as plt
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(veg[:,0],veg[:,1],c="black",lw=2,label="vegitation (deciduous)")
    ax.plot(soil[:,0],soil[:,1],c="gray",lw=1,label="soil")
    ax.plot(cloud[:,0],cloud[:,1],c="black",ls="dashed",label="cloud (water)")
    ax.plot(snow_med[:,0],snow_med[:,1],c="gray",ls="dashed",label="snow (medium granular)")
    ax.plot(water[:,0],water[:,1],c="gray",ls="-.",label="water")
    ax.plot(clear_sky[:,0],clear_sky[:,1],c="gray",ls="dotted",label="clear sky")
    for i in range(0,len(valexp)):
        ax.plot(ave_band,malbedo[i,:],"+",label=valexp[i])
    plt.xlim(0.4,1.5)
    plt.legend(bbox_to_anchor=(1.1, 0.3))
#    plt.show()

