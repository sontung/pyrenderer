from scipy.spatial.transform import Rotation as rot_mat_compute
import pyrr.matrix44
import numpy as np
from math import radians


def make_rotation_matrix(degrees, homo=True):
    axes = ["x", "y", "z"]
    rot_mat = np.identity(3, np.float32)
    for d_id, degree in enumerate(degrees):
        if degree != 0:
            rot_mat = rot_mat @ rot_mat_compute.from_euler(axes[d_id], degree, degrees=True).as_matrix()
    if homo:
        return to_homogeneous_matrix(rot_mat)
    return rot_mat


def to_homogeneous_matrix(mat):
    res = np.hstack([mat, np.zeros((3, 1))])
    res = np.vstack([res, np.zeros((4,))])
    res[3][3] = 1.0
    return res


def make_translation_matrix(moves):
    res = np.identity(4, np.float32)
    res[:3, 3] = moves
    return res


def make_scale_matrix(scales):
    res = np.identity(4, np.float32)
    res[0, 0] = scales[0]
    res[1, 1] = scales[1]
    res[2, 2] = scales[2]
    return res


def make_transformation_matrix(transforms):
    """
    create a transformation matrix 4x4
    :param transforms: {'scale': [2, 4, 2], 'rotation': [0, 90, 0]}
    :return:
    """
    res = np.identity(4, np.float32)
    if "position" in transforms:
        trans_mat = make_translation_matrix(transforms["position"])
        res = res @ trans_mat
    if "rotation" in transforms:
        rotation_mat = make_rotation_matrix(transforms["rotation"])
        res = res @ rotation_mat
    if "scale" in transforms:
        scale_mat = make_scale_matrix(transforms["scale"])
        res = res @ scale_mat
    return res


if __name__ == '__main__':
    tr = {
        "rotation": [
            0,
            90,
            90
        ]
    }
    mat = make_transformation_matrix(tr)
    vec = np.array([0, 1, 0, 1])
    mat3 = pyrr.matrix44.create_from_eulers([radians(90), radians(90), radians(0)])
    print(mat3 @ vec)
    print(vec @ mat3)
    u = vec[0]*mat3[:, 0] + vec[1]*mat3[:, 1] + vec[2]*mat3[:, 2] + vec[3]*mat3[:, 3]
    print(u)
    print(mat3)