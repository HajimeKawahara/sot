import numpy as np
import io_surface_type 
import matplotlib.pyplot as plt
import healpy as hp

nclass=3

cmap=io_surface_type.read_classification("../../../data/global_2008_5min.asc")
fig = plt.figure(figsize=(10,5))
ax=fig.add_subplot(111)
a=ax.imshow(cmap,cmap="tab20b")
plt.colorbar(a)
plt.title("MODIS CLASSIFICATION MAP 2008")
plt.show()
plt.close()

if nclass==4:
    cmap,vals,valexp=io_surface_type.merge_to_4classes(cmap)
    c4map,nside=io_surface_type.copy_to_healpix(cmap,nside=32)
    np.savez("cmap4class",c4map,vals,valexp)
    
    hp.mollview(c4map, title="test",flip="geo",cmap=plt.cm.Paired)
    hp.graticule(color="white")
    plt.savefig("cmap4class.pdf")
    plt.show()
if nclass==3:
    cmap,vals,valexp=io_surface_type.merge_to_3classes(cmap)
    c3map,nside=io_surface_type.copy_to_healpix(cmap,nside=32)
    np.savez("cmap3class",c3map,vals,valexp)
    
    hp.mollview(c3map, title="test",flip="geo",cmap=plt.cm.Paired)
    hp.graticule(color="white")
    plt.savefig("cmap3class.pdf")
    plt.show()
