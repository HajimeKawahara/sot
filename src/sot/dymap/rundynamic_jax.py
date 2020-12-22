"""
Summary
------------------
Core functions for Dynamic SOT compatible to JAX

"""

import jax.numpy as jnp
from jax import jit

@jit
def Mean_DYSOT(W,KS,KT,alpha,lc,Pid):
    Ni,Nj=jnp.shape(W)
    Kw=alpha*KT*(W@KS@W.T)
    IKw=jnp.eye(Ni)+Pid@Kw
    Xlc=jnp.linalg.solve(IKw,Pid@lc)
    Aast=alpha*KT@(W.T*Xlc).T@KS
    return Aast
