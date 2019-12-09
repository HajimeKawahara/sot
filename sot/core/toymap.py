import numpy as np
import io_surface_type


def make_multiband_map(cmap, refsurfaces, sky, vals, bands, onsky=False):
    nbands = np.shape(bands)[0]
    ncomp = len(refsurfaces)
    mmap = np.zeros((len(cmap), nbands))  # multiband map

    if len(refsurfaces) != len(vals):
        print("inconsisitent val and refsurces. CHECK IT.")

    # multiband nap
    malbedo = []
    for ibands in range(0, nbands):
        waves = bands[ibands][0]
        wavee = bands[ibands][1]
        malbedo_band = io_surface_type.set_meanalbedo(
            waves, wavee, refsurfaces, sky, onsky)
        malbedo.append(malbedo_band)
        for i in range(0, ncomp):
            mask = (cmap == vals[i])
            mmap[mask, ibands] = malbedo_band[i]
    malbedo = np.array(malbedo).T

    return mmap, malbedo


def make_ecmap(cmap, vals):
    ncomp = len(vals)
    ecmap = np.zeros((len(cmap), ncomp))  # map for each component
    for i in range(0, ncomp):
        mask = (cmap == vals[i])
        ecmap[mask, i] = 1.0
    return ecmap
