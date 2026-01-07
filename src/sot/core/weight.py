import healpy as hp
import jax.numpy as jnp
from jax import jit

def compute_omega(nside):
    omega = []
    npix = hp.nside2npix(nside)
    for ipix in range(0, npix):
        theta, phi = hp.pix2ang(nside, ipix)
        omega.append([theta, phi])
    return omega


def unit_vector_eO(inc, Thetaeq):
    # (3)
    eO = jnp.array(
        [
            jnp.sin(inc) * jnp.cos(Thetaeq),
            -jnp.sin(inc) * jnp.sin(Thetaeq),
            jnp.cos(inc),
        ]
    )
    return eO


def unit_vector_eS(Thetaeq, Thetav):
    # (3,nsamp)
    eS = jnp.array(
        [jnp.cos(Thetav - Thetaeq), jnp.sin(Thetav - Thetaeq), jnp.zeros(len(Thetav))]
    )
    return eS


def unit_vector_eR(zeta, Phiv, omega):
    # (3,nsamp,npix)
    jnp.array([Phiv]).T
    costheta = jnp.cos(omega[:, 0])
    sintheta = jnp.sin(omega[:, 0])
    cosphiPhi = jnp.cos(omega[:, 1] + jnp.array([Phiv]).T)
    sinphiPhi = jnp.sin(omega[:, 1] + jnp.array([Phiv]).T)
    #    cosphiPhi=np.cos(omega[:,1]-np.array([Phiv]).T)
    #    sinphiPhi=np.sin(omega[:,1]-np.array([Phiv]).T)

    x = cosphiPhi * sintheta
    y = jnp.cos(zeta) * sinphiPhi * sintheta + jnp.sin(zeta) * costheta
    z = -jnp.sin(zeta) * sinphiPhi * sintheta + jnp.cos(zeta) * costheta
    eR = jnp.array([x, y, z])

    return eR

@jit
def compute_weight(zeta, inc, Thetaeq, Thetav, Phiv, omega_vector):
    """computes geometric weights  
    
    Args:
        zeta (_type_): _description_
        inc (_type_): _description_
        Thetaeq (_type_): _description_
        Thetav (_type_): _description_
        Phiv (_type_): _description_

    Returns:
        array: illuminated weight
        array: visible weight
    """
    eO = unit_vector_eO(inc, Thetaeq)
    eS = unit_vector_eS(Thetaeq, Thetav)
    eR = unit_vector_eR(zeta, Phiv, omega_vector)
    WV = jnp.einsum("ijk,i->jk", eR, eO)
    WV = jnp.where(WV < 0.0, 0.0, WV)
    WI = jnp.einsum("ijk,ij->jk", eR, eS)
    WI = jnp.where(WI < 0.0, 0.0, WI)
    return WI, WV
