import numpy as np
import healpy as hp

def rgbmap_from_cmap(cmap):
    """
    Convert a categorical cmap (npix,) into an RGB map (npix, 3 ).

    Args:
        cmap (array-like): input categorical map, shape (npix,)

    Raises:
        ValueError: If cmap does not have exactly 3 unique values.

    Returns:
        np.ndarray: RGB map, shape (npix, 3)
    """
    cmap = np.array(cmap)
    cmap_rgb = np.zeros([len(cmap), 3])
    vals = np.unique(cmap)
    if len(vals) != 3:
        raise ValueError("cmap must have exactly 3 unique values for RGB mapping.")
    for i in range(len(vals)):
        cmap_rgb[cmap == vals[i], i] = 1.0
    return cmap_rgb

def rgb_mollweide(rgb_map, nside, xsize=1200, coord="G", red_scale=1.0, green_scale = 0.85, blue_scale = 0.6, nest=False):
    """ Project an RGB Healpix rgb_map onto a Mollweide projection.

    Args:
        rgb_map (array-like): RGB healpix map of shape (npix, 3)
        nside (int): healpix nside parameter
        xsize (int, optional): output image width in pixels. Defaults to 1200.
        coord (str, optional): coordinate system for projection. Defaults to "G".
        red_scale (float, optional): scaling factor for red channel. Defaults to 1.0.
        green_scale (float, optional): scaling factor for green channel. Defaults to 0.85.
        blue_scale (float, optional): scaling factor for blue channel. Defaults to 0.6.
        nest (bool, optional): whether to use nested pixel ordering. Defaults to False.

    Returns:
        np.ndarray: RGBA image of the projected map.
    """
    rgb_map = np.asarray(rgb_map)
    npix = hp.nside2npix(nside)
    assert rgb_map.shape == (npix, 3)

    rgb = np.nan_to_num(rgb_map, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

    # robust normalize (per-channel)
    vmin = np.percentile(rgb, 1, axis=0)
    vmax = np.percentile(rgb, 99, axis=0)
    rgb = (rgb - vmin) / (vmax - vmin + 1e-30)
    rgb = np.clip(rgb, 0.0, 1.0)

    #red channel scaling
    rgb[:, 0] *= red_scale
    rgb[:, 0] = np.clip(rgb[:, 0], 0.0, 1.0)

    #green channel scaling
    rgb[:, 1] *= green_scale
    rgb[:, 1] = np.clip(rgb[:, 1], 0.0, 1.0)

    #blue channel scaling
    rgb[:, 2] *= blue_scale
    rgb[:, 2] = np.clip(rgb[:, 2], 0.0, 1.0)

    proj = hp.projector.MollweideProj(xsize=xsize, coord=coord)

    vec2pix = lambda x, y, z: hp.vec2pix(nside, x, y, z, nest=nest)

    # effective region mask 
    footprint = proj.projmap(np.ones(npix, dtype=float), vec2pix)
    valid = footprint > 0.5
    alpha = valid.astype(float)

    # 2) project RGB channels
    imgs = [proj.projmap(rgb[:, c], vec2pix) for c in range(3)]
    rgb_img = np.dstack(imgs)

    rgba = np.dstack([rgb_img, alpha])
    return rgba

