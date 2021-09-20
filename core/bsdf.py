from mathematics.constants import InvPi
from mathematics.samplers import cosine_sampling
from collections import namedtuple

Scatter = namedtuple("Scatter", ["direction", "scale"])


class BSDFLambertian:
    def __init__(self, data):
        self.rho = data['rho']

    def scatter(self):
        res = Scatter(cosine_sampling(), self.evaluate())
        return res

    def evaluate(self):
        return self.rho * InvPi

    def pdf(self):
        return InvPi


class BSDF:
    def __init__(self, _type, data):
        self._type = _type
        if _type == "lambert":
            self.distribution = BSDFLambertian(data)
        else:
            print(f"[WARNING] bsdf of type {_type} not implemented")

    def scatter(self):
        return self.distribution.scatter()