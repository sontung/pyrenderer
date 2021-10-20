from .vec3_taichi import Vector4, Vector
from .constants import EPS
from .affine_transformation import to_homogeneous_matrix
from taichi_glsl.vector import normalize, dot, cross

import taichi as ti
import numpy as np


@ti.func
def rotate_to(vector):
    """
    transformation matrix to rotate the Y-axis to vector
    :param vector:
    :return:
    """
    vector = normalize(vector)
    res = ti.Vector.field(n=4, dtype=ti.f32, shape=(4, 1))

    if abs(vector[1]-1.0) < EPS:
        res[0][0] = Vector4(1.0, 0.0, 0.0, 0.0)
        res[1][0] = Vector4(0.0, 1.0, 0.0, 0.0)
        res[2][0] = Vector4(0.0, 0.0, 1.0, 0.0)
        res[3][0] = Vector4(0.0, 0.0, 0.0, 1.0)

    elif abs(vector[1]+1.0) < EPS:
        res[0][0] = Vector4(1.0, 0.0, 0.0, 0.0)
        res[1][0] = Vector4(0.0, -1.0, 0.0, 0.0)
        res[2][0] = Vector4(0.0, 0.0, 1.0, 0.0)
        res[3][0] = Vector4(0.0, 0.0, 0.0, 1.0)

    else:
        x = normalize(cross(vector, Vector(0.0, 1.0, 0.0)))
        z = normalize(cross(x, vector))
        res = ti.Vector.field(n=4, dtype=ti.f32, shape=(4, 1))
        res[0][0] = Vector4(x[0], x[1], x[2], 0.0)
        res[1][0] = Vector4(vector[0], vector[1], vector[2], 0.0)
        res[2][0] = Vector4(z[0], z[1], z[2], 0.0)
        res[3][0] = Vector4(0.0, 0.0, 0.0, 1.0)
    return res


@ti.func
def rotate_vector(mat, vec):
    homo_vec = Vector4(vec[0], vec[1], vec[2], 1.0)
    out_dir = Vector4(dot(homo_vec, mat[0][0]),
                      dot(homo_vec, mat[1][0]),
                      dot(homo_vec, mat[2][0]))
    out_dir = normalize(out_dir)
    return out_dir


@ti.func
def transpose(mat):
    res = ti.Vector.field(n=4, dtype=ti.f32, shape=(4, 1))
    r1 = mat[0][0]
    r2 = mat[1][0]
    r3 = mat[2][0]
    r4 = mat[3][0]
    res[0][0] = Vector4(r1[0], r2[0], r3[0], r4[0])
    res[1][0] = Vector4(r1[1], r2[1], r3[1], r4[1])
    res[2][0] = Vector4(r1[2], r2[2], r3[2], r4[2])
    res[3][0] = Vector4(r1[3], r2[3], r3[3], r4[3])
    return res
