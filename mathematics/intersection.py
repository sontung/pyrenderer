from .constants import EPS, MAX_F
from .fast_op import fast_dot3, cross_product, cross_product_vectorized, fast_subtract, fast_subtract_vectorized
import numpy as np
from numba import njit, guvectorize


def triangle_ray_intersection(vertices, ray):
    ret = {"origin": ray.position, "hit": False, "t": 0.0}

    p0 = vertices[0]
    p1 = vertices[1]
    p2 = vertices[2]
    e1 = p1-p0
    e2 = p2-p0
    q = np.cross(ray.direction, e2)
    a = np.dot(e1, q)
    if np.abs(a) < EPS:
        return ret
    f = 1.0/a
    s = ray.position-p0
    u = f*np.dot(s, q)
    if u < 0.0:
        return ret
    r = np.cross(s, e1)
    t = f*np.dot(e2, r)

    if t > ray.bounds[1] or t < EPS:
        return ret

    v = f*np.dot(ray.direction, r)
    if v < 0.0 or u+v > 1.0:
        return ret

    ray.bounds[1] = t
    ret["t"] = t
    ret["position"] = ray.position+t*ray.direction
    ret["hit"] = True
    # ret.normal = (1.0-u-v)*v_0.normal+u*v_1.normal+v*v_2.normal
    return ret


def triangle_ray_intersection_wo_cross(ray, a, e2r, sq, rdr):
    ret = {"hit": False, "t": 0.0}

    if -EPS < a < EPS:
        return ret
    f = 1.0/a
    t = f*e2r
    if t > ray.bounds[1] or t < EPS:
        return ret

    u = f*sq
    if u < 0.0:
        return ret

    v = f*rdr
    if v < 0.0 or u+v > 1.0:
        return ret

    ray.bounds[1] = t
    ret["t"] = t
    ret["origin"] = ray.position
    ret["position"] = ray.position+t*ray.direction
    ret["hit"] = True
    # ret.normal = (1.0-u-v)*v_0.normal+u*v_1.normal+v*v_2.normal
    return ret


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


@njit("f8[:], f8[:], f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:],")
def triangle_ray_intersection_grouping_numba(u1, u2, ray_bound, nb_triangles,
                                             s_array, q_array, r_array, p0_array,
                                             e1_array, e2_array, a_array, e2r_array, sq_array, rdr_array, res_holder):

    fast_subtract(u1, p0_array, s_array)
    cross_product(u2, e2_array, q_array)
    cross_product(s_array, e1_array, r_array)
    fast_dot3(e1_array, q_array, a_array)
    fast_dot3(e2_array, r_array, e2r_array)
    fast_dot3(s_array, q_array, sq_array)
    fast_dot3(u2, r_array, rdr_array)

    for i in range(nb_triangles):
        triangle_ray_intersection_numba(i, ray_bound, a_array[i], e2r_array[i], sq_array[i], rdr_array[i], res_holder)


# @profile
def triangle_ray_intersection_grouping(ray, nb_triangles, s_array, q_array, r_array, p0_array,
                                       e1_array, e2_array, a_array, e2r_array, sq_array, rdr_array, res_holder):
    try:
        u1 = ray.position_tile[nb_triangles]
        u2 = ray.direction_tile[nb_triangles]
    except KeyError:
        u1 = np.tile(ray.position, nb_triangles)
        u2 = np.tile(ray.direction, nb_triangles)
        ray.position_tile[nb_triangles] = u1
        ray.direction_tile[nb_triangles] = u2

    triangle_ray_intersection_grouping_numba(u1, u2, ray.bounds,
                                             nb_triangles,
                                             s_array, q_array, r_array, p0_array,
                                             e1_array, e2_array, a_array, e2r_array,
                                             sq_array, rdr_array, res_holder
                                             )
    results = []
    tmin = MAX_F
    for i in range(nb_triangles):
        if res_holder[i*2] > 0.0 and res_holder[i*2+1] < tmin:
            tmin = res_holder[i*2+1]
            ret = dict()
            ret["t"] = res_holder[i*2+1]
            ret["origin"] = ray.position
            ret["position"] = ray.position + res_holder[i*2+1] * ray.direction
            ret["hit"] = True
            results.append((ret, i))

    return results
