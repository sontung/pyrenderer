from .vec3_taichi import Vector4, Vector
from .constants import EPS
from taichi_glsl.vector import normalize, dot, cross
from taichi_glsl import mat

import taichi as ti


@ti.func
def rotate_to(vector):
    """
    transformation matrix to rotate the Y-axis to vector
    :param vector:
    :return:
    """
    vector = normalize(vector)

    res1 = Vector4(1.0, 0.0, 0.0, 0.0)
    res2 = Vector4(0.0, 1.0, 0.0, 0.0)
    res3 = Vector4(0.0, 0.0, 1.0, 0.0)
    res4 = Vector4(0.0, 0.0, 0.0, 1.0)

    if abs(vector[1]-1.0) < EPS:
        res1 = Vector4(1.0, 0.0, 0.0, 0.0)
        res2 = Vector4(0.0, 1.0, 0.0, 0.0)
        res3 = Vector4(0.0, 0.0, 1.0, 0.0)
        res4 = Vector4(0.0, 0.0, 0.0, 1.0)

    elif abs(vector[1]+1.0) < EPS:
        res1 = Vector4(1.0, 0.0, 0.0, 0.0)
        res2 = Vector4(0.0, -1.0, 0.0, 0.0)
        res3 = Vector4(0.0, 0.0, 1.0, 0.0)
        res4 = Vector4(0.0, 0.0, 0.0, 1.0)
    else:
        x = normalize(cross(vector, Vector(0.0, 1.0, 0.0)))
        z = normalize(cross(x, vector))
        res1 = Vector4(x[0], x[1], x[2], 0.0)
        res2 = Vector4(vector[0], vector[1], vector[2], 0.0)
        res3 = Vector4(z[0], z[1], z[2], 0.0)
        res4 = Vector4(0.0, 0.0, 0.0, 1.0)
    return res1, res2, res3, res4


@ti.func
def rotate_z_to(vector):
    """
    transformation matrix to rotate the Y-axis to vector
    :param vector:
    :return:
    """
    res1, res2, res3, res4 = rotate_to(vector)
    return res1, res3, res2, res4


@ti.func
def rotate_vector(res1, res2, res3, vec):
    out_dir = vec[0]*res1+vec[1]*res2+vec[2]*res3
    homo_vec = Vector(out_dir[0], out_dir[1], out_dir[2])
    homo_vec = normalize(homo_vec)
    return homo_vec


@ti.func
def transpose(m):
    res = ti.Vector.field(n=4, dtype=ti.f32, shape=(4, 1))
    r1 = m[0][0]
    r2 = m[1][0]
    r3 = m[2][0]
    r4 = m[3][0]
    res[0][0] = Vector4(r1[0], r2[0], r3[0], r4[0])
    res[1][0] = Vector4(r1[1], r2[1], r3[1], r4[1])
    res[2][0] = Vector4(r1[2], r2[2], r3[2], r4[2])
    res[3][0] = Vector4(r1[3], r2[3], r3[3], r4[3])
    return res
