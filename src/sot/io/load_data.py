import numpy as np
import healpy as hp

DATA_DIR = "data/"
REFDATA_DIR = "data/refdata/"


def get_data_filepath(filename, datadir=DATA_DIR):
    """get the full path of the data file

    Args:
        filename (str): filename of the test data
        datadir (str): directory of the test data

    Returns:
        str: full path of the test data file

    """
    from importlib.resources import files

    return files("sot").joinpath(datadir + filename)


def load_multi_kmap(file_path):
    """load the multi-kmap data
    Args:
        file_path (str): path to the data file
    Returns:
        tuple: cmap, nclass, npix_orig, nside_orig, vals, valexp
            kmap: multi k map
            nclass: number of classes (Nk)
            npix_orig: number of pixels in the original map
            nside_orig: nside of the original map
            dict_class (dict) dictionary of the label and values of the classes
    """
    dataclass = np.load(file_path)
    kmap = dataclass["arr_0"]
    nclass = len(np.unique(kmap))
    npix_orig = len(kmap)
    nside_orig = hp.npix2nside(npix_orig)
    dict_class = dict(zip(dataclass["arr_2"], dataclass["arr_1"]))
    return kmap, nclass, npix_orig, nside_orig, dict_class


def load_refdata():
    """load the reference data
    Returns:
        dict: dictionary of the reference data
            key: name of the reference data
            value: reflectivity data
    """

    # from astrobio/ipynb/reflectivity.ipynb
    def get_data_reffilepath(path):
        return get_data_filepath(path, REFDATA_DIR)

    cloud = np.loadtxt(get_data_reffilepath("clouds.txt"))
    cloud_ice = np.loadtxt(get_data_reffilepath("clouds_ice.txt"))
    snow_fine = np.loadtxt(get_data_reffilepath("fine_snow.txt"))
    snow_granular = np.loadtxt(get_data_reffilepath("granular_snow.txt"))
    snow_med = np.loadtxt(get_data_reffilepath("medium_snow.txt"))
    soil = np.loadtxt(get_data_reffilepath("soil.txt"))
    veg = np.loadtxt(get_data_reffilepath("veg_deciduous.txt"))
    ice = np.loadtxt(get_data_reffilepath("ice.txt"))
    #    water = np.loadtxt(get_data_reffilepath( "water.txt"))
    water = np.loadtxt(get_data_reffilepath("ocean_McLinden.csv"))

    # /sotica/bluedot/testlibradtran/test> python ../../iouvspec.py -i UVSPEC_CLEAR.INP
    cs = np.load(get_data_reffilepath("clear_sky.npz"))
    clear_sky, water, snow_fine, snow_granular, snow_med, soil, veg = (
        _normalize_reflection_data(
            snow_fine, snow_granular, snow_med, soil, veg, water, cs
        )
    )

    refdict = {
        "cloud": cloud,
        "cloud_ice": cloud_ice,
        "snow_fine": snow_fine,
        "snow_granular": snow_granular,
        "snow_med": snow_med,
        "soil": soil,
        "vegetation": veg,
        "ice": ice,
        "water": water,
        "clear_sky": clear_sky,
    }

    return refdict


def _normalize_reflection_data(
    snow_fine, snow_granular, snow_med, soil, veg, water, cs
):
    clear_sky = cs["arr_0"].T
    clear_sky[:, 0] = clear_sky[:, 0] / 1000.0  # nm->micron
    water[:, 0] = water[:, 0] / 1000.0
    snow_fine[:, 1] = snow_fine[:, 1] / 100.0
    snow_granular[:, 1] = snow_granular[:, 1] / 100.0
    snow_med[:, 1] = snow_med[:, 1] / 100.0
    soil[:, 1] = soil[:, 1] / 100.0
    veg[:, 1] = veg[:, 1] / 100.0
    return clear_sky, water, snow_fine, snow_granular, snow_med, soil, veg


def get_meanalbedo(ref, waves, wavee):
    mask = (ref[:, 0] >= waves) * (ref[:, 0] <= wavee)
    return np.mean(ref[mask, 1])
