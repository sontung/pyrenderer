from .constants import EPS
import numpy as np


def triangle_ray_intersection(vertices, ray):
    v_0 = vertices[0]
    v_1 = vertices[1]
    v_2 = vertices[2]
    e1 = v_1-v_0
    e2 = v_2-v_0

    ret = {"origin": ray.position, "hit": False, "t": 0.0,
           "position": np.array([0.0, 0.0, 0.0])}

    s = ray.position-v_0
    cross_e1_d = np.cross(e1, ray.dir)

    det = np.dot(cross_e1_d, e2)
    if np.abs(det) <= EPS:
        return ret

    f = 1.0/det
    cross_s_e2 = np.cross(s, e2)
    dot_s_e2_e1 = np.dot(cross_s_e2, e1)
    if np.abs(dot_s_e2_e1) <= EPS:
        return ret

    t = -f*dot_s_e2_e1
    if t >= ray.bounds[1] or t <= EPS:
        return ret

    u = -f*np.dot(cross_s_e2, ray.dir)
    if u < EPS or u > 1.0:
        return ret
    v = f*np.dot(cross_e1_d, s)
    if v < EPS or u + v > 1.0:
        return ret

    ray.bounds[1] = t
    ret["distance"] = t
    ret["position"] = ray.position+t*ray.dir
    ret["hit"] = True
    # ret.normal = (1.0-u-v)*v_0.normal+u*v_1.normal+v*v_2.normal
    return ret
