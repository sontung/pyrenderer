from .vec3 import normalize_vector, to_homogeneous_vector
from .constants import EPS
from .affine_transformation import to_homogeneous_matrix

import numpy as np


def rotate_to(vector):
    """
    transformation matrix to rotate the Y-axis to vector
    :param vector:
    :return:
    """
    vector = normalize_vector(vector)
    if abs(vector[1]-1.0) < EPS:
        return np.identity(4, np.float64)
    elif abs(vector[1]+1.0) < EPS:
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
    else:
        x = normalize_vector(np.cross(vector, np.array([0.0, 1.0, 0.0])))
        z = normalize_vector(np.cross(x, vector))
        res = np.vstack([x, vector, z])
        return to_homogeneous_matrix(res)


def rotate_vector(mat, vec):
    out_dir = to_homogeneous_vector(vec) @ mat
    out_dir = out_dir[:3]
    out_dir = normalize_vector(out_dir)
    return out_dir


def rotate_z_to(vector):
    """
    transformation matrix to rotate the Z-axis to vector
    :param vector:
    :return:
    """
    y = rotate_to(vector)
    _y = y[1]
    _z = y[2]
    y[1] = _z
    y[2] = -_y
    return y


def divide(mat):
    for i in range(mat.shape[0]):
        mat[i] = mat[i]/mat[i, 3]


def normalize(mat):
    for i in range(mat.shape[0]):
        mat[i] = normalize_vector(mat[i])


def transform(mat, trans_mat):
    for i in range(mat.shape[0]):
        mat[i] = mat[i] @ trans_mat
