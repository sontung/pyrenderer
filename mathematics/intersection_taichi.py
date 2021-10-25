import taichi as ti
from .vec3_taichi import *
import core.ray_taichi as ray
from core.bsdf_taichi import Materials
import random
import numpy as np
from math import isclose
from .constants import MAX_F, EPS
from accelerators.bvh_taichi import BVH
from taichi_glsl.randgen import randInt


@ti.func
def is_front_facing(ray_direction, normal):
    return ray_direction.dot(normal) < 0.0


@ti.func
def hit_sphere(center, radius, ray_origin, ray_direction, t_min, t_max):
    ''' Intersect a sphere of given radius and center and return
        if it hit and the least root. '''
    oc = ray_origin - center
    a = ray_direction.norm_sqr()
    half_b = oc.dot(ray_direction)
    c = (oc.norm_sqr() - radius**2)
    discriminant = (half_b**2) - a * c

    hit = discriminant >= 0.0
    root = -1.0
    if hit:
        sqrtd = discriminant**0.5
        root = (-half_b - sqrtd) / a

        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                hit = False

    return hit, root


@ti.func
def max_dim(v):
    dim = 0
    if v[0] > v[1]:
        if v[0] > v[2]:
            dim = 0
        else:
            dim = 2
    elif v[1] > v[2]:
        dim = 1
    else:
        dim = 2
    return dim


@ti.func
def permute(p, x, y, z):
    res = Vector(0.0, 0.0, 0.0)

    for i in ti.static(range(3)):
        if i == x:
            res[0] = p[i]
        elif i == y:
            res[1] = p[i]
        elif i == z:
            res[2] = p[i]

    return res


@ti.func
def ray_triangle_hit(v0, v1, v2, ro, rd, t0, t1):
    t = MAX_F
    hit = 0
    e1 = v1-v0
    e2 = v2-v0
    cross_e1_d = e1.cross(rd)
    det = cross_e1_d.dot(e2)
    if abs(det) > 0.0:
        f = 1.0/det
        s = ro - v0
        cross_s_e2 = s.cross(e2)
        dot_s_e2_e1 = cross_s_e2.dot(e1)
        t = -f*dot_s_e2_e1

        if t0 < t < t1:
            u = -f * cross_s_e2.dot(rd)
            if 0.0 <= u <= 1.0:
                v = f * cross_e1_d.dot(s)
                if v >= 0.0 and 1.0-u-v >= 0.0:
                    hit = 1

    return hit, t


@ti.func
def ray_triangle_hit2(p0, p1, p2, ro, rd, tMin, tMax):
    hit = 1
    t = MAX_F
    p0t = p0 - ro
    p1t = p1 - ro
    p2t = p2 - ro

    kz = max_dim(ti.abs(rd))
    kx = kz + 1
    if kx == 3:
        kx = 0
    ky = kx + 1
    if ky == 3:
        ky = 0
    d = permute(rd, kx, ky, kz)
    p0t = permute(p0t, kx, ky, kz)
    p1t = permute(p1t, kx, ky, kz)
    p2t = permute(p2t, kx, ky, kz)
    Sx = -d[0] / d[2]
    Sy = -d[1] / d[2]
    Sz = 1.0 / d[2]
    p0t[0] += Sx * p0t[2]
    p0t[1] += Sy * p0t[2]
    p1t[0] += Sx * p1t[2]
    p1t[1] += Sy * p1t[2]
    p2t[0] += Sx * p2t[2]
    p2t[1] += Sy * p2t[2]

    e0 = p1t[0] * p2t[1] - p1t[1] * p2t[0]
    e1 = p2t[0] * p0t[1] - p2t[1] * p0t[0]
    e2 = p0t[0] * p1t[1] - p0t[1] * p1t[0]

    if e0 == 0.0 or e1 == 0.0 or e2 == 0.0:
        p2txp1ty = ti.cast(p2t[0], ti.f64) * ti.cast(p1t[1], ti.f64)
        p2typ1tx = ti.cast(p2t[1], ti.f64) * ti.cast(p1t[0], ti.f64)
        e0 = ti.cast(p2typ1tx - p2txp1ty, ti.f32)
        p0txp2ty = ti.cast(p0t[0], ti.f64) * ti.cast(p2t[1], ti.f64)
        p0typ2tx = ti.cast(p0t[1], ti.f64) * ti.cast(p2t[0], ti.f64)
        e1 = ti.cast(p0txp2ty - p0typ2tx, ti.f32)
        p1txp0ty = ti.cast(p1t[0], ti.f64) * ti.cast(p0t[1], ti.f64)
        p1typ0tx = ti.cast(p1t[1], ti.f64) * ti.cast(p0t[0], ti.f64)
        e2 = ti.cast(p1typ0tx - p1txp0ty, ti.f32)

    if (e0 < 0.0 or e1 < 0.0 or e2 < 0.0) and (e0 > 0.0 or e1 > 0.0 or e2 > 0.0):
        hit = 0
    else:
        det = e0 + e1 + e2
        if abs(det) <= 0.0001:
            hit = 0
        else:
            p0t[2] *= Sz
            p1t[2] *= Sz
            p2t[2] *= Sz
            tScaled = e0 * p0t[2] + e1 * p1t[2] + e2 * p2t[2]
            if det < 0.0 and (tScaled >= 0.0 or tScaled < tMax * det):
                hit = 0
            elif det > 0.0 and (tScaled <= 0.0 or tScaled > tMax * det):
                hit = 0
            else:
                invDet = 1.0 / det
                t = tScaled * invDet

                # barycentric coordinates (not needed for now)
                # b0 = e0 * invDet
                # b1 = e1 * invDet
                # b2 = e2 * invDet
    return hit, t


class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
        self.id = -1
        self.box_min = [
            self.center[0] - radius, self.center[1] - radius,
            self.center[2] - radius
        ]
        self.box_max = [
            self.center[0] + radius, self.center[1] + radius,
            self.center[2] + radius
        ]

    @property
    def bounding_box(self):
        return self.box_min, self.box_max


BRANCH = 1.0
LEAF = 0.0


@ti.data_oriented
class World:
    def __init__(self):
        self.primitives = []
        self.lights = []

    @ti.func
    def sample_a_light(self):
        point = Vector(0.0, 0.0, 0.0)
        if len(self.lights) > 1:
            light_id = randInt(0, len(self.lights)-1)
            for i in ti.static(range(len(self.lights))):
                if i == light_id:
                    point = self.lights[i].sample_a_point()
        else:
            point = self.lights[0].sample_a_point()
        return point

    def add(self, prim):
        prim.id = len(self.primitives)
        self.primitives.append(prim)
        if prim.bsdf.emitting_light:
            self.lights.append(prim)

    def commit(self):
        ''' Commit should be called after all objects added.
            Will compile bvh and materials. '''
        self.n = len(self.primitives)
        self.materials = Materials(self.n)
        self.bvh = BVH(self.primitives)
        self.bvh.build()
        assert len(self.lights) > 0, "There is no lights!!!"

    def bounding_box(self, i):
        return self.bvh_min(i), self.bvh_max(i)

    @ti.func
    def hit_all(self, ray_origin, ray_direction):
        ''' Intersects a ray against all objects. '''
        hit_anything = False
        t_min = 0.0001
        closest_so_far = 99999.9
        hit_index = 0
        p = Point(0.0, 0.0, 0.0)
        normal = Vector(0.0, 0.0, 0.0)
        emissive = 0
        attenuation = Vector(0.0, 0.0, 0.0)
        scattered_dir = Vector(0.0, 0.0, 0.0)
        front_facing = True
        sided = 1
        curr = self.bvh.bvh_root

        # walk the bvh tree
        while curr != -1:
            obj_id, left_id, right_id, next_id = self.bvh.get_full_id(curr)

            if obj_id != -1:
                for i in ti.static(range(len(self.primitives))):
                    if i == obj_id:
                        hit, t, n, next_ray_d, att, emit, bsdf_sided = self.primitives[i].hit(ray_origin, ray_direction,
                                                                                              t_min, closest_so_far)
                        if hit > 0:
                            hit_anything = True
                            closest_so_far = t
                            hit_index = obj_id
                            normal = n
                            emissive = emit
                            attenuation = att
                            scattered_dir = next_ray_d
                            sided = bsdf_sided
                curr = next_id
            else:
                if self.bvh.hit_aabb(curr, ray_origin, ray_direction, t_min, closest_so_far):
                    # add left and right children
                    if left_id != -1:
                        curr = left_id
                    elif right_id != -1:
                        curr = right_id
                    else:
                        curr = next_id
                else:
                    curr = next_id

        if hit_anything:
            p = ray.at(ray_origin, ray_direction, closest_so_far)
            if not sided:
                front_facing = is_front_facing(ray_direction, normal)
                normal = normal if front_facing else -normal
        return hit_anything, closest_so_far, p, normal, front_facing, hit_index, emissive, attenuation, scattered_dir

    @ti.func
    def hit_slow(self, ray_origin, ray_direction):
        ''' Intersects a ray against all objects. '''
        hit_anything = False
        t_min = EPS
        closest_so_far = 99999.9
        hit_index = 0
        p = Point(0.0, 0.0, 0.0)
        normal = Vector(0.0, 0.0, 0.0)
        emissive = 0
        attenuation = Vector(0.0, 0.0, 0.0)
        scattered_dir = Vector(0.0, 0.0, 0.0)
        front_facing = True
        sided = 1

        for i in ti.static(range(len(self.primitives))):
            hit, t, n, next_ray_d, att, emit, bsdf_sided = self.primitives[i].hit(ray_origin, ray_direction,
                                                                                  t_min, closest_so_far)
            if hit > 0:
                hit_anything = True
                closest_so_far = t
                hit_index = i
                normal = n
                emissive = emit
                attenuation = att
                scattered_dir = next_ray_d
                sided = bsdf_sided

        if hit_anything and not sided:
            p = ray.at(ray_origin, ray_direction, closest_so_far)
            front_facing = is_front_facing(ray_direction, normal)
            normal = normal if front_facing else -normal
        return hit_anything, closest_so_far, p, normal, front_facing, hit_index, emissive, attenuation, scattered_dir
