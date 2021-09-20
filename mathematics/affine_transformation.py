from scipy.spatial.transform import Rotation as rot_mat_compute
import numpy as np


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
                "position": [
                    0,
                    2,
                    0
                ],
                "scale": [
                    2,
                    4,
                    2
                ],
                "rotation": [
                    0,
                    0,
                    -180
                ]
            }
    make_transformation_matrix(tr)