import random
import numpy as np
from math import sqrt, cos, sin
from .vec3 import normalize_vector
from .constants import Pi, PiOver2, PiOver4, InvPi


def concentric_sample_disk(u):
    u_offset = 2.0*u-np.array([1.0, 1.0])
    if u_offset[0] == 0 and u_offset[1] == 0:
        res = np.array([0.0, 0.0])
    else:
        if np.abs(u_offset[0]) > np.abs(u_offset[1]):
            r = u_offset[0]
            theta = PiOver4 * (u_offset[1] / u_offset[0])
        else:
            r = u_offset[1]
            theta = PiOver2 - PiOver4*(u_offset[0]/u_offset[1])
        res = r*np.array([np.cos(theta), np.sin(theta)])
    return res


def cosine_sample_hemisphere(normal_world_space):
    u = np.random.rand(2)
    d = concentric_sample_disk(u)
    z = np.sqrt(max([0.0, 1-d[0]*d[0]-d[1]*d[1]]))
    vec = np.array([d[0], d[1], z])
    r1, r2, r3, r4 = rotate_z_to(normal_world_space)
    return rotate_vector(r1, r2, r3, vec)


def cosine_hemisphere_pdf(cos_theta):
    return cos_theta*InvPi


def rotate_to(vector):
    """
    transformation matrix to rotate the Y-axis to vector
    :param vector:
    :return:
    """
    vector = normalize_vector(vector)

    res1 = np.array([1.0, 0.0, 0.0, 0.0])
    res2 = np.array([0.0, 1.0, 0.0, 0.0])
    res3 = np.array([0.0, 0.0, 1.0, 0.0])
    res4 = np.array([0.0, 0.0, 0.0, 1.0])

    if abs(vector[1]-1.0) < 0.0:
        res1 = np.array([1.0, 0.0, 0.0, 0.0])
        res2 = np.array([0.0, 1.0, 0.0, 0.0])
        res3 = np.array([0.0, 0.0, 1.0, 0.0])
        res4 = np.array([0.0, 0.0, 0.0, 1.0])

    elif abs(vector[1]+1.0) < 0.0:
        res1 = np.array([1.0, 0.0, 0.0, 0.0])
        res2 = np.array([0.0, -1.0, 0.0, 0.0])
        res3 = np.array([0.0, 0.0, 1.0, 0.0])
        res4 = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        x = normalize_vector(np.cross(vector, np.array([0.0, 1.0, 0.0])))
        z = normalize_vector(np.cross(x, vector))
        res1 = np.array([x[0], x[1], x[2], 0.0])
        res2 = np.array([vector[0], vector[1], vector[2], 0.0])
        res3 = np.array([z[0], z[1], z[2], 0.0])
        res4 = np.array([0.0, 0.0, 0.0, 1.0])
    return res1, res2, res3, res4


def rotate_z_to(vector):
    """
    transformation matrix to rotate the Y-axis to vector
    :param vector:
    :return:
    """
    res1, res2, res3, res4 = rotate_to(vector)
    return res1, res3, -res2, res4


def rotate_vector(res1, res2, res3, vec):
    out_dir = vec[0]*res1+vec[1]*res2+vec[2]*res3
    homo_vec = np.array([out_dir[0], out_dir[1], out_dir[2]])
    homo_vec = normalize_vector(homo_vec)
    return homo_vec
