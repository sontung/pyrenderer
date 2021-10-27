from mathematics.constants import InvPi
from mathematics.samplers import cosine_sampling
from mathematics.vec3_taichi import Vector
from collections import namedtuple
from mathematics.samplers import cosine_sample_hemisphere
from taichi_glsl.vector import vec2
from taichi_glsl.randgen import rand
import numpy as np
import taichi as ti

Scatter = namedtuple("Scatter", ["direction", "scale"])


@ti.func
def same_hemisphere(w, wp):
    return w[2]*wp[2] > 0


@ti.data_oriented
class BSDFLambertian:
    """
    implement lambertian reflection in
    https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Reflection_Functions#
    """
    def __init__(self, data):
        self.rho = Vector(data['albedo'][0], data['albedo'][1], data['albedo'][2])
        self.emitting_light = 0
        self.sided = 0

    @ti.func
    def scatter(self, wo):
        u = vec2(rand(), rand())
        wi = cosine_sample_hemisphere(u)
        if wo[2] < 0.0:
            wi[2] *= -1.0
        pdf = self.pdf(wo, wi)
        return wi, self.evaluate(), pdf

    @ti.func
    def evaluate(self):
        return self.rho

    @ti.func
    def pdf(self, wo, wi):
        res = 0.0
        if same_hemisphere(wo, wi):
            res = ti.abs(wi[2]) * InvPi
        return res


@ti.data_oriented
class BSDFLight:
    def __init__(self, data):
        self.rho = data['albedo']
        self.emitting_light = 1
        self.sided = 1

    @ti.func
    def evaluate(self):
        return Vector(self.rho, self.rho, self.rho)

    @ti.func
    def pdf(self, wo, wi):
        res = 0.0
        if same_hemisphere(wo, wi):
            res = ti.abs(wi[2]) * InvPi
        return res

    @ti.func
    def scatter(self, wo):
        u = vec2(rand(), rand())
        wi = cosine_sample_hemisphere(u)
        if wo[2] < 0.0:
            wi[2] *= -1.0
        pdf = self.pdf(wo, wi)
        return wi, self.evaluate(), pdf


@ti.data_oriented
class BSDF:
    def __init__(self, data):
        self._type = data["type"]
        if self._type == "lambert":
            self.distribution = BSDFLambertian(data)
        elif self._type == "null":
            self.distribution = BSDFLight(data)
        else:
            print(f"[WARNING] bsdf of type {self._type} not implemented")
            raise NotImplementedError
        self.emitting_light = self.distribution.emitting_light
        self.sided = self.distribution.sided

    def get_distribution(self):
        return self.distribution

    @ti.func
    def bsdf_info(self):
        return self.emitting_light, self.sided

    @ti.func
    def scatter(self):
        return self.distribution.scatter()