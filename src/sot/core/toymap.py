"""generates toy maps
"""

import numpy as np


def _compute_mean_albedo(ref, waves, wavee):
    mask = (ref[:, 0] >= waves) * (ref[:, 0] <= wavee)
    return np.mean(ref[mask, 1])


def generate_multiband_map(
    cmap, dict_class, refdict, reference_surfaces, bands, onsky=False, sky=None
):
    """generate a multiband map
    Args:
        cmap (np.ndarray): classification map
        dict_class (dict): dictionary of class names and their values
        refdict (dict): dictionary of reference data
        reference_surfaces (list): list of reference surfaces
        bands (list): list of bands, each band is a tuple of (start, end)
        onsky (bool): whether to include sky albedo
        sky (np.ndarray): sky albedo data
    Returns:
        mmap (np.ndarray): multiband map (npix, ncomp)
        spectrum_matrix (np.ndarray): spectrum matrix (ncomp, nbands)
    
    """
    nbands = np.shape(bands)[0]
    ncomp = len(reference_surfaces)
    if ncomp != len(dict_class):
        raise ValueError("inconsisitent numbers of dict_class and reference_sarfaces")

    # map
    labels = list(dict_class.keys())
    ncomp = len(labels)
    value2col = {dict_class[label]: i for i, label in enumerate(labels)}
    col_idx = np.vectorize(value2col.get)(cmap)
    mmap = np.eye(ncomp, dtype=int)[col_idx]

    # spectrum
    spectrum_matrix=[]
    for ibands in range(0, nbands):
        waves = bands[ibands][0]
        wavee = bands[ibands][1]
        if onsky:
            atm = _compute_mean_albedo(sky, waves, wavee)
        else:
            atm = 0.0
        
        ma = []
        for label in labels:
            raw_spectrum = refdict[reference_surfaces[label]]
            ma.append(_compute_mean_albedo(raw_spectrum, waves, wavee) + atm)
        spectrum_matrix.append(np.array(ma))

    spectrum_matrix = np.array(spectrum_matrix).T
    
    return mmap, spectrum_matrix

