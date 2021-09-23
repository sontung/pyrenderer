from math import sqrt
import numpy as np


def norm_squared(vector):
    return vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]


def norm(vector):
    return sqrt(norm_squared(vector))


def normalize_vector(vector):
    n = norm(vector)
    return vector/n


def to_homogeneous_vector(vector):
    res = np.ones((4,), np.float64)
    res[:3] = vector
    return res
