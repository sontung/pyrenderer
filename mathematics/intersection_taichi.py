import taichi as ti
from .vec3_taichi import *
import core.ray_taichi as ray
from core.bsdf_taichi import Materials
import random
import numpy as np
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
        prim.id = len(self.spheres)
        self.spheres.append(prim)
        if prim.bsdf.emitting_light:
            self.lights.append(prim)

    def commit(self):
        ''' Commit should be called after all objects added.
            Will compile bvh and materials. '''
        self.n = len(self.spheres)
        self.materials = Materials(self.n)
        self.bvh = BVH(self.spheres)
        self.bvh.build()
        assert len(self.lights) > 0, "There is no lights!!!"

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
                for i in ti.static(range(len(self.spheres))):
                    if i == obj_id:
                        hit, t, n, next_ray_d, att, emit, bsdf_sided = self.spheres[i].hit(ray_origin, ray_direction)
                        if hit > 0 and t_min < t < closest_so_far:
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

        if hit_anything and not sided:
            p = ray.at(ray_origin, ray_direction, closest_so_far)
            front_facing = is_front_facing(ray_direction, normal)
            normal = normal if front_facing else -normal
        return hit_anything, closest_so_far, p, normal, front_facing, hit_index, emissive, attenuation, scattered_dir

    @ti.func
    def scatter(self, ray_direction, p, n, front_facing, index):
        ''' Get the scattered direction for a ray hitting an object '''
        return self.materials.scatter(index, ray_direction, p, n, front_facing)
