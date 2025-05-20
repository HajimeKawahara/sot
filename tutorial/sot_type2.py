# %% [markdown]
# # Spin-Orbit Unmixing using jaxinv (Type 2)

# %% [markdown]
# last update: 5/8 (2025) Hajime Kawahara
# 
# In this tutorial, we use the Type 2 Bayesian linear inverse problem solver from jaxinv to estimate maps and spectra from multiband time-series data.

# %%

# %% [markdown]
# Note: It is known that the current implementation does not work properly unless 64-bit mode is enabled.

# %%
from jax import config
config.update("jax_enable_x64", True)


# %%
import numpy as np
import healpy as hp
import matplotlib
from sot.core import sepmat
from sot.core.weight import compute_weight 



# %%
fontsize=16
matplotlib.rcParams.update({'font.size':fontsize})
np.random.seed(53)

#set geometry                                                               
inc=45.0/180.0*np.pi
Thetaeq=np.pi
zeta=23.4/180.0*np.pi
Pspin=23.9344699/24.0 #Pspin: a sidereal day                                
wspin=2*np.pi/Pspin
Porb=365.242190402
worb=2*np.pi/Porb                                                          \

Ni=256
obst=np.linspace(0.0,Porb,Ni)

# %%
from sot.io.load_data import get_data_filepath
from sot.io.load_data import load_multi_kmap
import matplotlib.pyplot as plt

#multi k map
file_path = get_data_filepath("cmap3class.npz")
cmap, nclass, npix, nside, dict_class = load_multi_kmap(file_path)
Nk = len(dict_class)
print(dict_class)

hp.mollview(cmap, title="multi k map",flip="geo",cmap=plt.cm.Paired)
hp.graticule(color="white")


# %%
from sot.io.load_data import load_refdata

refdict = load_refdata()
reference_surfaces = {"ocean": "water", "desert": "soil", "veg": "vegetation"}
bands = [
    [0.4, 0.45],
    [0.45, 0.5],
    [0.5, 0.55],
    [0.55, 0.6],
    [0.6, 0.65],
    [0.65, 0.7],
    [0.7, 0.75],
    [0.75, 0.8],
    [0.8, 0.85],
    [0.85, 0.9],
]

from sot.core.toymap import generate_multiband_map
map_matrix, spectral_matrix = generate_multiband_map(
    cmap,
    dict_class,
    refdict,
    reference_surfaces,
    bands,
    onsky=True,
    sky=refdict["clear_sky"],
)

Nl = len(bands)
print(Nl)

# %%
import jax.numpy as jnp
from jaxinv.model.linear import type2
# geometric weight
Thetav = worb * obst
Phiv = np.mod(wspin * obst, 2 * np.pi)
WI, WV = compute_weight(nside, zeta, inc, Thetaeq, Thetav, Phiv)
W = jnp.array(WV[:, :] * WI[:, :])
Ni, Nj = np.shape(W)
map_matrix = jnp.array(map_matrix)
spectral_matrix = jnp.array(spectral_matrix)
# Light curve

#lc = jnp.sum(W * movmap, axis=1)
lc = type2(W, spectral_matrix, map_matrix)
noiselevel = 0.001
sigma = noiselevel * np.mean(lc)
noise = sigma * np.random.normal(0.0, 1.0, np.shape(lc))
lc = lc + noise


# %%
import jax.numpy as jnp
from jaxinv.bayes.gpkernel import rbf
from jaxinv.bayes.gpkernel import matern32
from sot.plot.plotdymap import plotseqmap
from jaxinv.bayes.meanmap import meanmap_inverse_type2

gamma = 1.0 / 180.0 * np.pi 
separation_matrix = sepmat.calc_sepmatrix(nside)
KS = rbf(separation_matrix, gamma)

#map k-direction covariance
KX = np.eye(Nk)

alpha = 0.25

Pid = jnp.eye(Ni*Nl) * sigma**-2 #when using jnp.eye Matrix becomes singular when x32
    
Mast = meanmap_inverse_type2(W, spectral_matrix, KS, KX, alpha, Pid, lc)
print(Mast.shape)

hp.mollview(Mast[:,0], title="k=0",flip="geo")
hp.graticule(color="white")
plt.savefig("k0.png", dpi=300)

hp.mollview(Mast[:,1], title="k=1",flip="geo")
hp.graticule(color="white")
plt.savefig("k1.png", dpi=300)

hp.mollview(Mast[:,2], title="k=2",flip="geo")
hp.graticule(color="white")
plt.savefig("k2.png", dpi=300)

lc_pred = type2(W, spectral_matrix, Mast)

fig = plt.figure(figsize=(20, 10))
#list_select = range(Nl)
list_select = [0, 9]

ax = fig.add_subplot(211)
for i in list_select:
    plt.plot(obst, lc[:, i], ".", label="band %d" % (i + 1), alpha=1.0, color = "C"+str(i))
    #plt.plot(obst, lc_pred[:, i], label="pred band %d" % (i + 1), alpha=0.5, color = "C"+str(i))

ax = fig.add_subplot(212)
for i in list_select:
    plt.plot(obst, lc[:, i], ".", label="band %d" % (i + 1), alpha=1.0, color = "C"+str(i))
    plt.plot(obst, lc_pred[:, i], label="pred band %d" % (i + 1), alpha=0.5, color = "C"+str(i))

plt.xlabel("Time [days]")
plt.ylabel("Intensity [arbitrary unit]")
plt.title("Light curve")
plt.legend()
plt.savefig("lc.png", dpi=300)


# %%
