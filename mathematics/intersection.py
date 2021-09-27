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
def triangle_ray_intersection_grouping(ray, triangles, e1e2, cross_holder, cross_a, cross_b,
                                       s_holder):
    all_s = np.zeros((len(triangles), 3), np.float64)
    for idx, vertices in enumerate(triangles):
        p0 = vertices[0]
        e1 = vertices[3][0]
        e2 = vertices[3][1]
        fast_subtract2(ray.position, p0, s_holder)
        cross_a[idx*6: idx*6+3] = ray.direction
        cross_a[idx*6+3: idx*6+6] = s_holder
        cross_b[idx*6: idx*6+3] = e2
        cross_b[idx*6+3: idx*6+6] = e1
        all_s[idx] = s_holder
    cross_product2(cross_a, cross_b, cross_holder)
    all_crosses = cross_holder.reshape((-1, 3))
    # v = cross_product(np.hstack(cross_a), np.hstack(cross_b))
    a_e2r = fast_dot3(e1e2, all_crosses.reshape((e1e2.shape[0],)))
    results = [triangle_ray_intersection_wo_cross(ray,
                                                  all_crosses[i],
                                                  all_crosses[i+1],
                                                  a_e2r[i], a_e2r[i+1], all_s[i//2],
                                                  )
               for i in range(0, len(triangles)*2, 2)]
    return results
