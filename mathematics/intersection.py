from .constants import EPS
from .fast_op import fast_dot3, fast_subtract2, cross_product2, fast_dot
import numpy as np


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


# @profile
def triangle_ray_intersection_wo_cross(ray, q, r, a, e2r, s):
    ret = {"hit": False, "t": 0.0}

    if -EPS < a < EPS:
        return ret
    f = 1.0/a

    t = f*e2r
    if t > ray.bounds[1] or t < EPS:
        return ret

    u = f*fast_dot(s, q)
    if u < 0.0:
        return ret

    v = f*fast_dot(ray.direction, r)
    if v < 0.0 or u+v > 1.0:
        return ret

    ray.bounds[1] = t
    ret["t"] = t
    ret["origin"] = ray.position
    ret["position"] = ray.position+t*ray.direction
    ret["hit"] = True
    # ret.normal = (1.0-u-v)*v_0.normal+u*v_1.normal+v*v_2.normal
    return ret


# @profile
def triangle_ray_intersection_grouping(ray, triangles, q_array, r_array, p0_array,
                                       e1_array, e2_array):
    nb_triangles = len(triangles)
    res = np.zeros((nb_triangles*3,), np.float64)
    try:
        u1 = ray.position_tile[nb_triangles]
        u2 = ray.direction_tile[nb_triangles]
    except KeyError:
        u1 = np.tile(ray.position, nb_triangles)
        u2 = np.tile(ray.direction, nb_triangles)
        ray.direction_tile[nb_triangles] = u2
        ray.position_tile[nb_triangles] = u1
    fast_subtract2(u1, p0_array, res)
    all_s = res.reshape((nb_triangles, 3))

    cross_product2(u2, e2_array, q_array)
    cross_product2(res, e1_array, r_array)
    a_array = fast_dot3(e1_array, q_array)
    e2r_array = fast_dot3(e2_array, r_array)

    q_array = q_array.reshape((-1, 3))
    r_array = r_array.reshape((-1, 3))

    results = [triangle_ray_intersection_wo_cross(ray,
                                                  q_array[i],
                                                  r_array[i],
                                                  a_array[i], e2r_array[i], all_s[i],
                                                  )
               for i in range(nb_triangles)]
    return results
