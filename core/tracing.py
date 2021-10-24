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
    def __init__(self, world, depth, img_w, img_h):
        self.world = world
        self.depth = depth
        self.r_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(img_w, img_h, depth))
        self.e_field = ti.field(dtype=ti.f32, shape=(img_w, img_h, depth))
        self.dr_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(img_w, img_h, depth))
        self.att_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(img_w, img_h, depth))

    @ti.func
    def reset(self):
        for i in range(self.depth):
            self.dr_field[i, 0] = Vector(0.0, 0.0, 0.0)
            self.att_field[i, 0] = Vector(0.0, 0.0, 0.0)
            self.idr_field[i, 0] = Vector(0.0, 0.0, 0.0)

    @ti.func
    def sample_direct_lighting(self, hit_pos, in_dir_world_space, scale):
        radiance = Vector(0.0, 0.0, 0.0)
        hit, t, hit_pos, normal, front_facing, index, emitting_light, emissive, scattered_dir = self.world.hit_all(
            hit_pos, in_dir_world_space)
        if hit > 0 and emitting_light > 0 and in_dir_world_space.dot(normal) < 0.0:
            radiance += scale * emissive
        return radiance

    @ti.func
    def trace(self, ro, rd, depth, x, y):
        hit_anything = 0
        max_bounce = 0
        for bounce in range(depth):
            hit, t, hit_pos, normal, front_facing, index, emitting_light, attenuation, scattered_dir = self.world.hit_all(
                ro, rd)
            max_bounce += 1
            if hit > 0 and emitting_light > 0:
                hit_anything = 1
                if rd.dot(normal) < 0.0:
                    self.r_field[x, y, bounce] = attenuation
                    self.e_field[x, y, bounce] = 1.0
                    break
                object_to_world1, object_to_world2, object_to_world3, object_to_world4 = rotate_to(normal)
                scattered_dir_world = normalize(
                    rotate_vector(object_to_world1, object_to_world2, object_to_world3, scattered_dir))
                ro = hit_pos
                rd = scattered_dir_world
            elif hit > 0 and bounce < depth-1:
                hit_anything = 1
                object_to_world1, object_to_world2, object_to_world3, object_to_world4 = rotate_to(normal)
                scattered_dir_world = normalize(
                    rotate_vector(object_to_world1, object_to_world2, object_to_world3, scattered_dir))

                # direct lighting
                light_sample = self.world.sample_a_light()
                dir_towards_light = normalize(light_sample - hit_pos)
                dr = self.sample_direct_lighting(hit_pos, dir_towards_light, attenuation)
                # dr = self.sample_direct_lighting(hit_pos, scattered_dir_world, attenuation)

                self.dr_field[x, y, bounce] = dr
                self.att_field[x, y, bounce] = attenuation

                # indirect
                ro = hit_pos
                # ro = offset_ray(hit_pos, normal)
                rd = scattered_dir_world

            elif hit == 0:
                break

        if hit_anything > 0:
            for bounce in range(1, max_bounce):
                bid = max_bounce - bounce - 1
                c1 = self.e_field[x, y, bid]*self.r_field[x, y, bid]
                if self.e_field[x, y, bid] < 0.5:
                    idr = self.att_field[x, y, bid]*self.r_field[x, y, bid+1]
                    c1 = self.dr_field[x, y, bid]+idr
                self.r_field[x, y, bid] = c1
        return self.r_field[x, y, 0]

