import taichi as ti
from .vec3_taichi import *
import core.ray_taichi as ray
from core.bsdf_taichi import Materials
import random
import numpy as np
from .constants import MAX_F, EPS
from accelerators.bvh_taichi import BVH


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
def ray_triangle_hit(p0, p1, p2, ro, rd):
    e1 = p1 - p0
    e2 = p2 - p0
    q = rd.cross(e2)
    a = e1.dot(q)
    t = MAX_F
    hit = 0
    if abs(a) >= EPS:
        f = 1.0/a
        s = ro-p0
        u = f*s.dot(q)
        if 1.0 >= u >= 0.0:
            r = s.cross(e1)
            v = f*rd.dot(r)
            if v >= 0.0 and u+v <= 1.0:
                hit = 1
                t = f * e2.dot(r)
    return hit, t


@ti.func
def ray_triangle_hit2(v0, v1, v2, ro, rd):
    u = v1 - v0
    v = v2 - v0
    norm = u.cross(v)
    depth = MAX_F * 2
    s, t = 0., 0.
    hit = 0

    b = norm.dot(rd)
    if abs(b) >= EPS:
        w0 = ro - v0
        a = -norm.dot(w0)
        r = a / b
        if r > 0:
            ip = ro + r * rd
            uu = u.dot(u)
            uv = u.dot(v)
            vv = v.dot(v)
            w = ip - v0
            wu = w.dot(u)
            wv = w.dot(v)
            D = uv * uv - uu * vv
            s = (uv * wv - vv * wu) / D
            t = (uv * wu - uu * wv) / D
            if 0 <= s <= 1:
                if 0 <= t and s + t <= 1:
                    depth = r
                    hit = 1
    return hit, depth


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
        self.spheres = []

    def add(self, sphere):
        sphere.id = len(self.spheres)
        self.spheres.append(sphere)

    def commit(self):
        ''' Commit should be called after all objects added.
            Will compile bvh and materials. '''
        self.n = len(self.spheres)
        self.materials = Materials(self.n)
        self.bvh = BVH(self.spheres)
        self.bvh.build()

    def bounding_box(self, i):
        return self.bvh_min(i), self.bvh_max(i)

    @ti.func
    def hit_all(self, ray_origin, ray_direction):
        ''' Intersects a ray against all objects. '''
        hit_anything = False
        t_min = EPS
        closest_so_far = 99999.9
        hit_index = 0
        p = Point(0.0, 0.0, 0.0)
        n = Vector(0.0, 0.0, 0.0)
        front_facing = True
        curr = self.bvh.bvh_root

        # walk the bvh tree
        while curr != -1:
            obj_id, left_id, right_id, next_id = self.bvh.get_full_id(curr)

            if obj_id != -1:
                for i in ti.static(range(len(self.spheres))):
                    if i == obj_id:
                        hit, t, n = self.spheres[i].hit(ray_origin, ray_direction)
                        if hit > 0 and t_min < t < closest_so_far:
                            hit_anything = True
                            closest_so_far = t
                            hit_index = obj_id
                curr = next_id
            else:
                if self.bvh.hit_aabb(curr, ray_origin, ray_direction, t_min,
                                     closest_so_far):
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
            front_facing = is_front_facing(ray_direction, n)
            n = n if front_facing else -n

        return hit_anything, closest_so_far, p, n, front_facing, hit_index

    @ti.func
    def scatter(self, ray_direction, p, n, front_facing, index):
        ''' Get the scattered direction for a ray hitting an object '''
        return self.materials.scatter(index, ray_direction, p, n, front_facing)
