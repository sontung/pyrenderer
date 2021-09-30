import jax.numpy as jnp
from jax import jit
from jax.ops import index_update
from jax import vmap


@jit
def jax_subtract(x, y):
    return x-y


def jax_subtract_vmap(x, y):
    return vmap(jax_subtract)(x, y)


jax_subtract_vmap_compiled = jit(jax_subtract_vmap)