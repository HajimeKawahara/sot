import numpy as np
import healpy as hp

DATA_DIR = "data/"

def get_data_filepath(filename):
    """get the full path of the data file

    Args:
        filename (str): filename of the test data
        dirname (str): directory name of the test data  (default: "data/testdata/")

    Returns:
        str: full path of the test data file

    """
    from importlib.resources import files

    return files("sot").joinpath(DATA_DIR + filename)

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

