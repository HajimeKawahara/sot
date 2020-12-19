import numpy as np
import os


def read_refdata(refdata):
    # from astrobio/ipynb/reflectivity.ipynb
    cloud = np.loadtxt(os.path.join(refdata, "clouds.txt"))
    cloud_ice = np.loadtxt(os.path.join(refdata, "clouds_ice.txt"))
    snow_fine = np.loadtxt(os.path.join(refdata, "fine_snow.txt"))
    snow_granular = np.loadtxt(os.path.join(refdata, "granular_snow.txt"))
    snow_med = np.loadtxt(os.path.join(refdata, "medium_snow.txt"))
    soil = np.loadtxt(os.path.join(refdata, "soil.txt"))
    veg = np.loadtxt(os.path.join(refdata, "veg_deciduous.txt"))
    ice = np.loadtxt(os.path.join(refdata, "ice.txt"))
#    water = np.loadtxt(os.path.join(refdata, "water.txt"))
    water = np.loadtxt(os.path.join(refdata, "ocean_McLinden.csv"))
    
    # /sotica/bluedot/testlibradtran/test> python ../../iouvspec.py -i UVSPEC_CLEAR.INP
    cs = np.load(os.path.join(refdata, "clear_sky.npz"))
    clear_sky = cs["arr_0"].T
    clear_sky[:, 0] = clear_sky[:, 0]/1000.0  # nm->micron
    water[:, 0] = water[:, 0]/1000.0

    snow_fine[:, 1] = snow_fine[:, 1]/100.0
    snow_granular[:, 1] = snow_granular[:, 1]/100.0
    snow_med[:, 1] = snow_med[:, 1]/100.0
    soil[:, 1] = soil[:, 1]/100.0
    veg[:, 1] = veg[:, 1]/100.0

    return cloud, cloud_ice, snow_fine, snow_granular, snow_med, soil, veg, ice, water, clear_sky


def get_meanalbedo(ref, waves, wavee):
    mask = (ref[:, 0] >= waves)*(ref[:, 0] <= wavee)
    return np.mean(ref[mask, 1])
