import random
import numpy as np
from math import sqrt, cos, sin
from .constants import Pi


def cosine_sampling():
    phi = random.random() * 2.0 * Pi
    cos_t = sqrt(random.random())

    sin_t = sqrt(1 - cos_t * cos_t)
    x = cos(phi) * sin_t
    z = sin(phi) * sin_t
    y = cos_t

    return np.array([x, y, z])
