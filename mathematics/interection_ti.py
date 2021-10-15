import taichi as ti
from .constants import EPS


@ti.func
def fast_subtract(x, y, s):
    for i in range(x.shape[0]):
        s[i] = x[i]-y[i]


@ti.func
def cross_product(x, y, res):
    shape = res.shape[0] // 3
    for i in range(shape):
        idx = i*3
        res[idx] = x[idx+1]*y[idx+2] - x[idx+2]*y[idx+1]
        res[idx+1] = x[idx+2]*y[idx] - x[idx]*y[idx+2]
        res[idx+2] = x[idx]*y[idx+1] - x[idx+1]*y[idx]


@ti.func
def fast_dot3(x, ty, res):
    for i in range(0, x.shape[0]//3):
        res[i] = x[i*3]*ty[i*3]+x[i*3+1]*ty[i*3+1]+x[i*3+2]*ty[i*3+2]


@ti.func
def triangle_ray_intersection_numba(ind, ray_bound, a, e2r,
                                    sq, rdr, res_holder):
    # if -EPS < a < EPS:
    #     res_holder[ind*2] = -1.0
    #     return -1
    f = 1.0/a
    t = f*e2r
    # if t > ray_bound[1] or t < EPS:
    #     res_holder[ind*2] = -1.0
    #     return -1
    #
    # u = f*sq
    # if u < 0.0:
    #     res_holder[ind*2] = -1.0
    #     return -1
    #
    # v = f*rdr
    # if v < 0.0 or u+v > 1.0:
    #     res_holder[ind*2] = -1.0
    #     return -1
    #
    # res_holder[ind*2] = 1.0
    # res_holder[ind*2+1] = t
    # ray_bound[1] = t
    return -1


@ti.kernel
def triangle_ray_intersection_grouping_numba(u1: ti.template(), u2: ti.template(), ray_bound: ti.template(),
                                             nb_triangles: ti.i32,
                                             s_array: ti.template(), q_array: ti.template(),
                                             r_array: ti.template(), p0_array: ti.template(),
                                             e1_array: ti.template(), e2_array: ti.template(),
                                             a_array: ti.template(), e2r_array: ti.template(),
                                             sq_array: ti.template(), rdr_array: ti.template(),
                                             res_holder: ti.template()):
    fast_subtract(u1, p0_array, s_array)
    cross_product(u2, e2_array, q_array)
    cross_product(s_array, e1_array, r_array)
    fast_dot3(e1_array, q_array, a_array)
    fast_dot3(e2_array, r_array, e2r_array)
    fast_dot3(s_array, q_array, sq_array)
    fast_dot3(u2, r_array, rdr_array)

    for i in range(nb_triangles):
        triangle_ray_intersection_numba(i, ray_bound, a_array[i], e2r_array[i], sq_array[i], rdr_array[i], res_holder)