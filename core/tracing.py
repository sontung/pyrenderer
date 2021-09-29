import numpy as np
import sys


# @profile
def ray_casting(ray, scene):
    ret = scene.hit_faster(ray)

    if not ret["hit"]:
        with open("debug/nothit.txt", "a") as afile:
            print(ray.position[0], ray.position[1], ray.position[2],
                  ray.direction[0], ray.direction[1], ray.direction[2], file=afile)
        return np.array([0.0, 0.0, 0.0])
    else:
        if ret["bsdf"].emitting_light:
            return ret["bsdf"].evaluate()
        return ret["bsdf"].rho


def ray_casting_delayed(ray, scene, a_register):
    ret = scene.hit_delayed(ray, a_register)