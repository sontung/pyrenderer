from .constants import EPS
import numpy as np


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

# @profile
def triangle_ray_intersection_wo_cross(vertices, ray, q, r):
    ret = {"origin": ray.position, "hit": False, "t": 0.0, "position": np.array([0.0, 0.0, 0.0])}

    p0 = vertices[0]
    # p1 = vertices[1]
    # p2 = vertices[2]
    e1 = vertices[3][0]
    e2 = vertices[3][1]

    a = np.dot(e1, q)
    if -EPS < a < EPS:
        return ret
    f = 1.0/a

    t = f*np.dot(e2, r)
    if t > ray.bounds[1] or t < EPS:
        return ret

    s = ray.position-p0
    u = f*np.dot(s, q)
    if u < 0.0:
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
def triangle_ray_intersection_grouping(ray, triangles):
    cross_a = []
    cross_b = []
    for vertices in triangles:
        p0 = vertices[0]
        p1 = vertices[1]
        p2 = vertices[2]
        e1 = p1 - p0
        e2 = p2 - p0
        s = ray.position - p0
        cross_a.extend([ray.direction, s])
        cross_b.extend([e2, e1])
    all_crosses = np.cross(cross_a, cross_b)
    results = [triangle_ray_intersection_wo_cross(triangles[i//2],
                                                  ray,
                                                  all_crosses[i],
                                                  all_crosses[i+1]) for i in range(0, len(triangles)*2, 2)]
    return results
