import numpy as np
from .constants import EPS
from math import cos, sin, radians


def euler2quaternion(angles):
    """
    create an unit quat for a given angle
    :param angles:
    :return:
    """
    if np.sum(np.abs(angles-np.array([0.0, 0.0, 180.0]))) <= EPS or \
            np.sum(np.abs(angles-np.array([180.0, 0.0, 0.0]))) <= EPS:
        return np.array([0.0, 0.0, -1.0, 0.0])
    c1 = cos(radians(angles[2]/2.0))
    c2 = cos(radians(angles[1]/2.0))
    c3 = cos(radians(angles[0]/2.0))
    s1 = sin(radians(angles[2]/2.0))
    s2 = sin(radians(angles[1]/2.0))
    s3 = sin(radians(angles[0]/2.0))
    x = c1 * c2 * s3 - s1 * s2 * c3
    y = c1 * s2 * c3 + s1 * c2 * s3
    z = s1 * c2 * c3 - c1 * s2 * s3
    w = c1 * c2 * c3 + s1 * s2 * s3
    return np.array([x, y, z, w])