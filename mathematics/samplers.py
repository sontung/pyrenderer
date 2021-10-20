import random
import numpy as np
from math import sqrt, cos, sin
from .constants import Pi
from .vec3_taichi import Vector
from taichi_glsl.randgen import rand
import taichi as ti


@ti.func
def cosine_sampling():
    phi = rand() * 2.0 * Pi
    cos_t = ti.sqrt(rand())

    sin_t = ti.sqrt(1 - cos_t * cos_t)
    x = ti.cos(phi) * sin_t
    z = ti.sin(phi) * sin_t
    y = cos_t

    return Vector(x, y, z)
