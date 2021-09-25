from .constants import EPS
import numpy as np


def cross(l, r):
    return np.array([l[1] * r[2] - l[2] * r[1], l[2] * r[0] - l[0] * r[2], l[0] * r[1] - l[1] * r[0]])


def triangle_ray_intersection(vertices, ray):
    ret = {"origin": ray.position, "hit": False, "t": 0.0, "position": np.array([0.0, 0.0, 0.0])}

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


def triangle_ray_intersection2(vertices, ray):
    v_0 = vertices[0]
    v_1 = vertices[1]
    v_2 = vertices[2]
    e1 = v_1-v_0
    e2 = v_2-v_0

    ret = {"origin": ray.position, "hit": False, "t": 0.0, "position": np.array([0.0, 0.0, 0.0])}
    s = ray.position-v_0

    # cross_e1_d, cross_s_e2 = np.cross([e1, s], [ray.direction, e2])
    cross_e1_d = cross(e1, ray.direction)
    cross_s_e2 = cross(s, e2)

    # print("cross", cross_e1_d, np.cross(e1, ray.direction), cross(e1, ray.direction))

    det = np.dot(cross_e1_d, e2)
    if np.abs(det) < EPS:
        print("reject det zero")
        return ret

    f = 1.0/det

    dot_s_e2_e1 = np.dot(cross_s_e2, e1)
    if np.abs(dot_s_e2_e1) < EPS:
        print("reject t zero")
        return ret

    t = -f*dot_s_e2_e1
    if t > ray.bounds[1] or t < EPS:
        print(f"reject t out bound {t} {ray.bounds[1]} {f} {dot_s_e2_e1}")
        return ret

    u = -f*np.dot(cross_s_e2, ray.direction)
    if u < EPS or u > 1.0:
        return ret
    v = f*np.dot(cross_e1_d, s)
    if v < EPS or u + v > 1.0:
        return ret

    ray.bounds[1] = t
    ret["t"] = t
    ret["position"] = ray.position+t*ray.direction
    ret["hit"] = True
    # ret.normal = (1.0-u-v)*v_0.normal+u*v_1.normal+v*v_2.normal
    return ret


def test():
    from core.ray import Ray
    ray = Ray(np.array([0.,  1.,  6.8]), np.array([ 0.32018365, -0.8,       -6.74333795]))
    tri = np.array( [[7.00118345e-01, 1.11022302e-16, 1.70214382e-01],
                     [1.30216309e-01, 5.55111512e-17, - 1.14109676e-04],
                     [1.30216309e-01,  6.00000000e-01, - 1.14109676e-04]])
    res = triangle_ray_intersection(tri, ray)
    print(res)
