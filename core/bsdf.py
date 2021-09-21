from mathematics.constants import InvPi
from mathematics.samplers import cosine_sampling
from collections import namedtuple

Scatter = namedtuple("Scatter", ["direction", "scale"])


class BSDFLambertian:
    def __init__(self, data):
        self.rho = data['albedo']

    def scatter(self):
        res = Scatter(cosine_sampling(), self.evaluate())
        return res

    def evaluate(self):
        return self.rho * InvPi

    def pdf(self):
        return InvPi


class BSDF:
    def __init__(self, data):
        self._type = data["type"]
        if self._type == "lambert":
            self.distribution = BSDFLambertian(data)
        else:
            print(f"[WARNING] bsdf of type {self._type} not implemented")
            raise NotImplementedError

    def scatter(self):
        return self.distribution.scatter()