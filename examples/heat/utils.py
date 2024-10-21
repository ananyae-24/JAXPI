import scipy.io

import jax
import jax.numpy as jnp
from jax import vmap
def get_dataset(T=2.0, L=1, c=2, n_t=200, n_x=128):
    t_star = jnp.linspace(0, T, n_t)
    x_star = jnp.linspace(0, L, n_x)

    u_exact_fn = lambda t, x: jnp.exp((-(jnp.pi*c)**2)*t)*jnp.sin(jnp.pi*x)

    u_exact = vmap(vmap(u_exact_fn, (None, 0)), (0, None))(t_star, x_star)

    return u_exact, t_star, x_star
