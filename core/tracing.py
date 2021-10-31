import numpy as np
import taichi as ti
from taichi_glsl.vector import normalize, invLength, dot, sqrLength
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
        # self.att_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(img_w, img_h, depth))
        self.cosine_field = ti.field(dtype=ti.f32, shape=(img_w, img_h, depth))

    @ti.func
    def sample_direct_lighting(self, p, normal):
        radiance = Vector(0.0, 0.0, 0.0)
        p2, n2, emissive = self.world.sample_a_light()

        w = normalize(p2 - p)
        w2 = normalize(p - p2)
        t_at_light = (p2[0] - p[0])/w[0]

        hit, t, d1, d2, d3, d4, d5, d6 = self.world.hit_all(
            p, w, 0.0001, t_at_light)
        if hit == 0:
            if dot(n2, w2) > 0.0:
                radiance += emissive
            # if dot(normal, w) > 0.0 and dot(n2, w2) > 0.0:
            #     radiance += emissive * dot(normal, w) * dot(n2, w2) / sqrLength(p - p2)
        return radiance

    @ti.func
    def get_normal(self, ro, rd, depth, x, y):
        hit, t, hit_pos, normal, front_facing, index, emitting_light, attenuation, scattered_dir = self.world.hit_all(
            ro, rd)
        return ti.abs(normal)

    @ti.func
    def trace2(self, ro, rd, depth, x, y):
        L = Vector(0.0, 0.0, 0.0)
        beta = Vector(1.0, 1.0, 1.0)
        for bounce in range(depth):
            hit, t, hit_pos, normal, emitting_light, attenuation, wi, pdf = self.world.hit_all(
                ro, rd, 0.00001, 99999.9)
            # print(hit, bounce, t, ro, rd, wi)
            # print(normal)
            if bounce == 0 and hit > 0 and emitting_light > 0:
                inv_rd = -rd
                if inv_rd.dot(normal) > 0.0:
                    L += beta*inv_rd.dot(normal)

            if hit == 0 or bounce >= depth:
                break

            # direct lighting
            Ld = beta * self.sample_direct_lighting(hit_pos, normal)
            L += Ld

            # indirect lighting
            # beta *= attenuation*ti.abs(wi.dot(normal))/pdf
            ro = hit_pos
            rd = wi
        return L

    @ti.func
    def trace(self, ro, rd, depth, x, y):
        hit_anything = 0
        max_bounce = 0
        for bounce in range(depth):
            hit, t, hit_pos, normal, front_facing, index, emitting_light, attenuation, scattered_dir = self.world.hit_all(
                ro, rd)
            # object_to_world1, object_to_world2, object_to_world3, object_to_world4 = rotate_to(normal)
            # scattered_dir_world = normalize(
            #     rotate_vector(object_to_world1, object_to_world2, object_to_world3, scattered_dir))
            # print(bounce, hit, ro, rd, t, normal, hit_pos)
            max_bounce += 1
            if hit > 0:
                object_to_world1, object_to_world2, object_to_world3, object_to_world4 = rotate_to(normal)
                scattered_dir = normalize(
                    rotate_vector(object_to_world1, object_to_world2, object_to_world3, scattered_dir))

                if emitting_light > 0:
                    hit_anything = 1
                    if rd.dot(normal) < 0.0:
                        self.r_field[x, y, bounce] = attenuation
                        self.e_field[x, y, bounce] = 1.0
                        break
                elif bounce < depth-1:
                    hit_anything = 1

                    # direct lighting
                    light_sample = self.world.sample_a_light()
                    dir_towards_light = normalize(light_sample - hit_pos)
                    dr = self.sample_direct_lighting(hit_pos, dir_towards_light, attenuation)
                    # dr = self.sample_direct_lighting(hit_pos, scattered_dir_world, attenuation)

                    self.dr_field[x, y, bounce] = dr
                    self.r_field[x, y, bounce] = attenuation
                    self.cosine_field[x, y, bounce] = scattered_dir.dot(normal)*invLength(normal)*invLength(scattered_dir)

                ro = hit_pos
                rd = scattered_dir

            else:
                break

        if hit_anything > 0:
            for bounce in range(1, max_bounce):
                bid = max_bounce - bounce - 1
                c1 = self.e_field[x, y, bid]*self.r_field[x, y, bid]
                if self.e_field[x, y, bid] < 0.5:
                    c1 = self.r_field[x, y, bid]*self.r_field[x, y, bid+1]*self.cosine_field[x, y, bid]
                    # c1 = self.dr_field[x, y, bid]+idr
                self.r_field[x, y, bid] = c1
        return self.r_field[x, y, 0] + self.dr_field[x, y, 0]

