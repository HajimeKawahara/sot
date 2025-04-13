"""reproduces Figure 2 in Kawahara (2016)."""

import jax.numpy as jnp

def modulation_factor_max(Theta, zeta, inc, Thetaeq):
    """computes the modulation factor at maximum weight, defined in Kawahara (2016)

    Args:
        Theta (float): phase angle (radian)
        zeta (float): obliquity angle (radian)
        inc (float)): inclination angle (radian)
        Thetaeq (float): phase angle at equinox (radian)

    Returns:
        float: modulation factor at maximum weight
    """

    fac1 = (
        -jnp.cos(zeta)
        + jnp.cos(inc) * jnp.sin(zeta) * jnp.sin(Theta - Thetaeq)
        - jnp.cos(zeta) * jnp.cos(Theta) * jnp.sin(inc)
    )
    fac2 = (
        jnp.cos(Theta - Thetaeq) ** 2
        + 2 * jnp.cos(Theta - Thetaeq) * jnp.cos(Thetaeq) * jnp.sin(inc)
        + jnp.cos(Thetaeq) ** 2 * jnp.sin(inc) ** 2
        + (
           jnp.cos(inc) * jnp.sin(zeta)
            - jnp.cos(zeta) * jnp.sin(Theta - Thetaeq)
            + jnp.cos(zeta) * jnp.sin(inc) * jnp.sin(Thetaeq)
        )
        ** 2
    )
    return fac1 / fac2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="max approx")
    parser.add_argument(
        "-i", nargs=1, default=[0.0], help="inclination [deg]", type=float
    )
    parser.add_argument("-t", nargs=1, default=[0.0], help="Thetaeq [deg]", type=float)
    args = parser.parse_args()

    N = 180
    gTheta = jnp.linspace(0, 2 * jnp.pi, 2 * N)
    gzeta = jnp.linspace(0, jnp.pi, N)
    X, Y = jnp.meshgrid(gTheta, gzeta)
    gridT = gTheta * jnp.array([jnp.ones(N)]).T
    gridz = (gzeta * jnp.array([jnp.ones(2 * N)]).T).T

    arr = modulation_factor_max(
        gridT, gridz, args.i[0] / 180 * jnp.pi, args.t[0] / 180 * jnp.pi
    )
    print(jnp.shape(arr))
    fig = plt.figure()
    plt.contour(
        X,
        Y,
        arr,
        colors="white",
        levels=[-1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25],
    )
    plt.imshow(
        arr, cmap="coolwarm", vmin=-1.0, vmax=1.0, extent=[0, 2 * jnp.pi, jnp.pi, 0]
    )
    plt.gca().invert_yaxis()
    plt.ylabel("$\zeta$", fontsize=15)
    plt.xlabel("$\Theta$", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.savefig("rott.png")
    plt.show()
