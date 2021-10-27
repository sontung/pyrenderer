import random
import numpy as np
from math import sqrt, cos, sin
from .constants import Pi, PiOver2, PiOver4, InvPi
from .vec3_taichi import Vector
from taichi_glsl.randgen import rand
from taichi_glsl.vector import vec2
import taichi as ti


@ti.func
def concentric_sample_disk(u):
    u_offset = 2.0*u-vec2(1, 1)
    res = vec2(0.0, 0.0)
    if u_offset[0] == 0 and u_offset[1] == 0:
        res = vec2(0.0, 0.0)
    else:
        theta = 0.0
        r = 0.0
        if ti.abs(u_offset[0]) > ti.abs(u_offset[1]):
            r = u_offset[0]
            theta = PiOver4 * (u_offset[1] / u_offset[0])
        else:
            r = u_offset[1]
            theta = PiOver2 - PiOver4*(u_offset[0]/u_offset[1])
        res = r*vec2(ti.cos(theta), ti.sin(theta))
    return res


@ti.func
def cosine_sample_hemisphere(u):
    d = concentric_sample_disk(u)
    z = ti.sqrt(ti.max(0.0, 1-d[0]*d[0]-d[1]*d[1]))
    return Vector(d[0], d[1], z)


@ti.func
def cosine_hemisphere_pdf(cos_theta):
    return cos_theta*InvPi


@ti.func
def cosine_sampling():
    phi = rand() * 2.0 * Pi
    cos_t = ti.sqrt(rand())

    sin_t = ti.sqrt(1 - cos_t * cos_t)
    x = ti.cos(phi) * sin_t
    z = ti.sin(phi) * sin_t
    y = cos_t

    return Vector(x, y, z)
