from math import sqrt


def norm_squared(vector):
    return vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]


def norm(vector):
    return sqrt(norm_squared(vector))


def normalize_vector(vector):
    n = norm(vector)
    return vector/n
