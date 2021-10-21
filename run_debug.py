import jax.numpy

from mathematics.constants import EPS, MAX_F
from mathematics.fast_op import fast_dot3, cross_product, fast_subtract, numba_tile, compute_pos
import numpy as np
from numba import njit
from jax import vmap, jit
from jax.ops import index_update, index


@njit("(i8, f8[:], f8, f8, f8, f8, f8[:])")
def triangle_ray_intersection_numba(ind, ray_bound, a, e2r, sq, rdr, res_holder):
    if -EPS < a < EPS:
        res_holder[ind*2] = -1.0
        return
    f = 1.0/a
    t = f*e2r
    if t > ray_bound[1] or t < EPS:
        res_holder[ind*2] = -1.0
        return

    u = f*sq
    if u < 0.0:
        res_holder[ind*2] = -1.0
        return

    v = f*rdr
    if v < 0.0 or u+v > 1.0:
        res_holder[ind*2] = -1.0
        return

    res_holder[ind*2] = 1.0
    res_holder[ind*2+1] = t
    ray_bound[1] = t

@vmap
def tri_jax(a, e2r, sq, rdr):
    f = 1.0/a
    t = f*e2r
    u = f*sq
    v = f*rdr
    return t
    # index_update(res_holder, ind, 1.0)
    # index_update(res_holder, ind, t)

@profile
def main():
    pos = np.random.rand(3)
    dir = np.random.rand(3)
    nb_triangles = 5000
    u1 = numba_tile(pos, nb_triangles)
    u2 = numba_tile(dir, nb_triangles)
    bounds = np.zeros((2,), np.float32)
    s_array = np.random.rand(nb_triangles * 3,)
    q_array = np.random.rand(nb_triangles * 3,)
    r_array = np.random.rand(nb_triangles * 3,)
    p0_array = np.random.rand(nb_triangles * 3,)
    e1_array = np.random.rand(nb_triangles * 3,)
    e2_array = np.random.rand(nb_triangles * 3,)
    a_array = np.random.rand(nb_triangles)
    e2r_array = np.zeros((nb_triangles,), np.float32)
    sq_array = np.zeros((nb_triangles,), np.float32)
    rdr_array = np.zeros((nb_triangles,), np.float32)
    res_holder = np.zeros((nb_triangles*2,), np.float32)

    fast_subtract(u1, p0_array, s_array)
    cross_product(u2, e2_array, q_array)
    cross_product(s_array, e1_array, r_array)
    fast_dot3(e1_array, q_array, a_array)
    fast_dot3(e2_array, r_array, e2r_array)
    fast_dot3(s_array, q_array, sq_array)
    fast_dot3(u2, r_array, rdr_array)

    for _ in range(100):
        for i in range(nb_triangles):
            triangle_ray_intersection_numba(i, bounds, a_array[i], e2r_array[i], sq_array[i], rdr_array[i], res_holder)

    a_array = jax.numpy.array(a_array)
    e2r_array = jax.numpy.array(e2r_array)
    sq_array = jax.numpy.array(sq_array)
    rdr_array = jax.numpy.array(rdr_array)

    for _ in range(100):
        g = tri_jax(a_array, e2r_array, sq_array, rdr_array)


main()
