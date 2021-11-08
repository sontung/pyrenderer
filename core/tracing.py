import numpy as np
import taichi as ti
from taichi_glsl.scalar import isnan
from taichi_glsl.vector import normalize, invLength, dot, sqrLength
from mathematics.vec3_taichi import Vector
from mathematics.mat4_taichi import rotate_to, rotate_vector, transpose
from mathematics.constants import EPS, Pi, InvPi
from mathematics.samplers import cosine_sample_hemisphere_convenient
from core.ray import Ray
import sys


@ti.func
def dot_or_zero(n, lv):
    return max(0.0, n.dot(lv))


@ti.func
def mis_power_heuristic(pf, pg):
    # Assume 1 sample for each distribution
    f = pf*pf
    g = pg*pg
    return f / (f + g)


@ti.func
def compute_area_light_pdf(t_light, ray_dir, light_normal, light_area):
    pdf = 0.0
    l_cos = light_normal.dot(-ray_dir)
    if l_cos > 1e-4:
        tmp = ray_dir * t_light
        dist_sqr = tmp.dot(tmp)
        pdf = dist_sqr / (light_area * l_cos)
    return pdf


@ti.func
def compute_brdf_pdf(normal, sample_dir):
    return dot_or_zero(normal, sample_dir) / np.pi


@ti.func
def offset_ray(ro, normal):
    ro_new = ro+normal*EPS
    return ro_new


class PathTracer:
    def __init__(self, world, depth, img_w, img_h):
        self.world = world
        self.depth = depth
        # self.beta_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(img_w, img_h))
        # self.e_field = ti.field(dtype=ti.f32, shape=(img_w, img_h, depth))
        # self.dr_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(img_w, img_h, depth))
        # self.cosine_field = ti.field(dtype=ti.f32, shape=(img_w, img_h, depth))

    @ti.func
    def sample_direct_lighting(self, hit_pos, hit_normal, hit_color):
        p2, n2, light_color = self.world.sample_a_light()

        direct_li = ti.Vector([0.0, 0.0, 0.0])
        fl = InvPi * hit_color * light_color
        light_pdf, brdf_pdf = 0.0, 0.0

        # sample area light
        to_light_dir = normalize(p2 - hit_pos)
        if to_light_dir.dot(hit_normal) > 0:
            hit, t, d1, d2, emitting_light, d3, d4, d5 = self.world.hit_all(
                hit_pos, to_light_dir, 0.00001, 9999.9)
            l_visible = emitting_light > 0
            if l_visible:
                light_pdf = compute_area_light_pdf(t, to_light_dir, n2, 1.0)
                brdf_pdf = compute_brdf_pdf(hit_normal, to_light_dir)
                if light_pdf > 0 and brdf_pdf > 0:
                    w = mis_power_heuristic(light_pdf, brdf_pdf)
                    nl = dot_or_zero(to_light_dir, hit_normal)
                    direct_li += fl * w * nl / light_pdf

        # sample brdf
        brdf_dir, brdf_pdf = cosine_sample_hemisphere_convenient(hit_normal)
        if brdf_pdf > 0:
            hit, t, d1, d2, emitting_light, d3, d4, d5 = self.world.hit_all(
                hit_pos, brdf_dir, 0.00001, 9999.9)
            l_visible = emitting_light > 0
            if l_visible:
                light_pdf = compute_area_light_pdf(t, brdf_dir, n2, 1.0)
                if light_pdf > 0:
                    w = mis_power_heuristic(brdf_pdf, light_pdf)
                    nl = dot_or_zero(brdf_dir, hit_normal)
                    direct_li += fl * w * nl / brdf_pdf
        return direct_li

    @ti.func
    def sample_direct_lighting2(self, p, normal):
        radiance = Vector(0.0, 0.0, 0.0)
        p2, n2, emissive = self.world.sample_a_light()

        w = normalize(p2 - p)
        w2 = normalize(p - p2)
        t_at_light = (p2[0] - p[0])/w[0]

        hit, t, d1, d2, d3, d4, d5, d6 = self.world.hit_all(
            p, w, 0.00001, t_at_light)
        if hit == 0:
            dot1 = dot(normal, w)
            dot2 = dot(n2, w2)
            if dot1 > 0.0 and dot2 > 0.0:
                radiance += emissive*dot1*dot2/sqrLength(p - p2)
        return radiance

    @ti.func
    def get_normal(self, ro, rd, depth, x, y):
        hit, t, hit_pos, normal, front_facing, index, emitting_light, attenuation, scattered_dir = self.world.hit_all(
            ro, rd)
        return ti.abs(normal)

    @ti.func
    def trace(self, ro, rd, depth, x, y):
        L = Vector(0.0, 0.0, 0.0)
        beta = Vector(1.0, 1.0, 1.0)
        for bounce in range(depth):
            if bounce >= depth:
                break

            hit, t, hit_pos, normal, emitting_light, attenuation, wi, pdf = self.world.hit_all(
                ro, rd, 0.00001, 99999.9)
            # print(hit, t, ro, rd, wi, attenuation)
            # print(normal)
            if hit > 0 and emitting_light > 0:
                inv_rd = -rd
                d1 = inv_rd.dot(normal)
                if d1 > 0.0:
                    if bounce == 0:
                        L += attenuation * beta
                    else:
                        L += attenuation*beta*d1
                    break
                else:
                    break

            if hit == 0:
                break

            # direct lighting
            if pdf > 0.0:
                beta *= InvPi*attenuation * dot_or_zero(normal, wi)/pdf
                # Ld = beta * self.sample_direct_lighting(hit_pos, normal, attenuation)
                Ld = beta * self.sample_direct_lighting2(hit_pos, normal)
                L += Ld

            ro = hit_pos
            rd = wi
        return L
