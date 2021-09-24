import numpy as np

@profile
def ray_casting(ray, scene):
    ret = scene.hit_faster(ray)
    ret2 = scene.hit(ray)

    if not ret["hit"]:
        return np.array([0.0, 0.0, 0.0])
    else:
        print("hit")
        return np.array([1.0, 0.0, 0.0])
