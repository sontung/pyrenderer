import numpy as np


def ray_casting(ray, scene):
    ret = scene.hit(ray)
    if not ret["hit"]:
        return np.array([0.0, 0.0, 0.0])
    else:
        return np.array([1.0, 0.0, 0.0])
