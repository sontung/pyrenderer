import numpy as np
import taichi as ti
from taichi_glsl.vector import normalize
from mathematics.vec3_taichi import Vector
from mathematics.mat4_taichi import rotate_to, rotate_vector, transpose
from mathematics.constants import EPS
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


@ti.func
def offset_ray(ro, normal):
    ro_new = ro+normal*EPS
    return ro_new


class PathTracer:
    def __init__(self, world):
        self.world = world

    @ti.func
    def sample_direct_lighting(self, hit_pos, in_dir_world_space, scale):
        radiance = Vector(0.0, 0.0, 0.0)
        hit, t, hit_pos, normal, front_facing, index, emitting_light, emissive, scattered_dir = self.world.hit_all(
            hit_pos, in_dir_world_space)
        if hit > 0 and emitting_light > 0 and in_dir_world_space.dot(normal) < 0.0:
            radiance += scale * emissive
        return radiance

    def sample_indirect_lighting(self, hit_info, scene, logged_rays):
        radiance = np.zeros((3,), np.float64)
        scatter = hit_info["bsdf"].scatter()
        in_dir = scatter.direction
        in_dir_world_space = rotate_vector(hit_info["object_to_world"], in_dir)
        new_ray = Ray(hit_info["position"], in_dir_world_space, hit_info["depth"]-1)
        res = path_tracing(new_ray, scene, logged_rays)
        reflection = res[1]
        radiance += scatter.scale * reflection
        return radiance

    @ti.func
    def trace(self, ro, rd, depth):
        radiance_e = Vector(0.0, 0.0, 0.0)
        radiance_r = Vector(0.0, 0.0, 0.0)
        while depth > 0:
            hit, t, hit_pos, normal, front_facing, index, emitting_light, attenuation, scattered_dir = self.world.hit_all(
                ro, rd)
            hit_pos = offset_ray(hit_pos, normal)
            if hit > 0:
                if emitting_light > 0:
                    if rd.dot(normal) < 0.0:
                        radiance_e += attenuation

                elif depth > 0:
                    # object_to_world = rotate_to(normal)
                    # world_to_object = transpose(object_to_world)
                    # out_dir = rotate_vector(world_to_object, ro - hit_pos)
                    # idr = sample_indirect_lighting(hit_info, scene)

                    light_sample = self.world.sample_a_light()
                    dir_towards_light = normalize(light_sample - hit_pos)
                    radiance_r += self.sample_direct_lighting(hit_pos, dir_towards_light, attenuation)
            depth = 0
            ro = hit_pos
            rd = scattered_dir
        return radiance_e, radiance_r

