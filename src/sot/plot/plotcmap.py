import matplotlib.pyplot as plt
from sot.utils.convert_maps import rgbmap_from_cmap
from sot.utils.convert_maps import rgb_mollweide
from healpy import npix2nside


def mollview_rgb(img, perm = [0, 1, 2], ax=None):
    """plot rgb image on mollweide projection
    
    Args:
        img (array): healpix array, (npix,) or (npix, 3)
        perm (list, optional): permutation of RGB channels. Defaults to [0, 1, 2].
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. Defaults to None.

    Raises:
        ValueError: Input image must be (npix,) or (npix, 3)
    """
    input_len = len(img.shape)
    if input_len == 1 :
        map_rgb = rgbmap_from_cmap(img)[:, perm]
    elif input_len == 2 and img.shape[1] == 3:
        map_rgb = img[:, perm]
    else:
        raise ValueError("Input image must be (npix,) or (npix, 3)")

    nside = npix2nside(map_rgb.shape[0])
    cmap_rgb_img = rgb_mollweide(map_rgb, nside)
    if ax is None:
        plt.imshow(cmap_rgb_img, origin="lower", interpolation="nearest")
        plt.axis("off")
        plt.gca().invert_xaxis()
    else:
        ax.imshow(cmap_rgb_img, origin="lower", interpolation="nearest")
        ax.axis("off")
        ax.invert_xaxis()

