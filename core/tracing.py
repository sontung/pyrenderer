import numpy as np
import sys

# @profile
def ray_casting(ray, scene):
    # v0 = [7.00118345e-01, 1.11022302e-16, 1.70214382e-01],
    # v1 = [1.30216309e-01, 5.55111512e-17, - 1.14109676e-04],
    # v2 = [1.30216309e-01,  6.00000000e-01, - 1.14109676e-04]
    # p = (np.array(v0)+np.array(v1)+np.array(v2))/3
    # p = p[0]
    # ray.direction = p - ray.position
    # print(p, ray.position, ray.direction)
    ret = scene.hit_faster(ray)
    # ray.reset_bounds()
    ret2 = scene.hit(ray)
    if ret["hit"] != ret2["hit"]:
        print("bvh", ret)
        print("bf", ret2)
        sys.exit()
    else:
        if abs(ret2["t"] - ret["t"]) > 0.001:
            print(ret2["t"], ret["t"])

    if not ret["hit"]:
        return np.array([0.0, 0.0, 0.0])
    else:
        return np.array([1.0, 0.0, 0.0])
