import numpy as np
from mathematics.mat4 import rotate_to, rotate_vector
from mathematics.vec3 import normalize_vector
from core.ray import Ray
import sys


# @profile
def ray_casting(ray, scene, normal_vis=True):
    ret = scene.hit_faster(ray)

    if not ret["hit"]:
        with open("debug/nothit.txt", "a") as afile:
            print(ray.position[0], ray.position[1], ray.position[2],
                  ray.direction[0], ray.direction[1], ray.direction[2], file=afile)
        return np.array([0.0, 0.0, 0.0])
    else:
        if ret["bsdf"].emitting_light:
            return ret["bsdf"].evaluate()
        if normal_vis:
            return np.abs(ret["normal"])
        return ret["bsdf"].rho


def sample_direct_lighting(hit_info, scene, logged_rays, in_dir_world_space=None):
    radiance = np.zeros((3,), np.float64)
    scatter = hit_info["bsdf"].scatter()
    if in_dir_world_space is None:
        in_dir = scatter.direction
        in_dir_world_space = rotate_vector(hit_info["object_to_world"], in_dir)
    new_ray = Ray(hit_info["position"], in_dir_world_space, 0)
    res = path_tracing(new_ray, scene, logged_rays)
    emissive = res[0]
    radiance += scatter.scale * emissive
    return radiance


def sample_indirect_lighting(hit_info, scene, logged_rays):
    radiance = np.zeros((3,), np.float64)
    scatter = hit_info["bsdf"].scatter()
    in_dir = scatter.direction
    in_dir_world_space = rotate_vector(hit_info["object_to_world"], in_dir)
    new_ray = Ray(hit_info["position"], in_dir_world_space, hit_info["depth"]-1)
    res = path_tracing(new_ray, scene, logged_rays)
    reflection = res[1]
    radiance += scatter.scale * reflection
    return radiance


def path_tracing(ray, scene, logged_rays=None):
    ret = scene.hit_faster(ray)
    null_val = np.array([0.0, 0.0, 0.0])

    if not ret["hit"]:
        return null_val, null_val

    if logged_rays is not None:
        if ret["bsdf"].emitting_light:
            logged_rays.add(ray, t=ret["t"], color=[0, 1, 0])
        else:
            logged_rays.add(ray, t=ret["t"], color=[1, 0, 0])
            # logged_rays.add_line(ret["position"], ret["position"]+ret["normal"]*1, color=[0, 0, 1])

    if ret["bsdf"].emitting_light:
        return ret["bsdf"].evaluate(), null_val

    if ray.depth == 0:
        return null_val, null_val

    object_to_world = rotate_to(ret["normal"])
    world_to_object = object_to_world.T
    out_dir = rotate_vector(world_to_object, ray.position - ret["position"])
    hit_info = {
        "object_to_world": object_to_world,
        "world_to_object": world_to_object,
        "out_dir": out_dir,
        "position": ret["position"],
        "normal": ret["normal"],
        "depth": ray.depth,
        "bsdf": ret["bsdf"]
    }

    light_sample = scene.sample_light()
    dir_towards_light = normalize_vector(light_sample-ret["position"])
    dr = sample_direct_lighting(hit_info, scene, logged_rays, in_dir_world_space=dir_towards_light)
    idr = sample_indirect_lighting(hit_info, scene, None)
    return null_val, dr+idr
