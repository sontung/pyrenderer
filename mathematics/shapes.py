import numpy as np
import trimesh
from .constants import MAX_F
import sys
from .intersection import triangle_ray_intersection
from .affine_transformation import make_transformation_matrix
from .bbox import BBox


class Quad:
    def __init__(self, trans_mat, bsdf):
        default_vertices = np.array([
            [-0.5, 0, -0.5],
            [0.5, 0, -0.5],
            [0.5, 0, 0.5],
            [-0.5, 0, 0.5],
        ])
        default_faces = np.array([
            [0, 1, 2],
            [2, 3, 0]
        ], np.uint)
        self.trans_mat = trans_mat
        self.mesh = trimesh.Trimesh(vertices=default_vertices,
                                    faces=default_faces,
                                    process=False)
        self.mesh.apply_transform(self.trans_mat)
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.bsdf = bsdf
        self.bounds = BBox(None, None)
        self.bounds.from_vertices(self.vertices)

    def visualize(self):
        self.mesh.show()

    def hit(self, ray):
        ret = {"origin": ray.position, "hit": False, "t": MAX_F,
               "position": np.array([0.0, 0.0, 0.0])}
        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            ret2 = triangle_ray_intersection([self.vertices[triangle[0]],
                                              self.vertices[triangle[1]],
                                              self.vertices[triangle[2]]], ray)
            if ret2["hit"] and ret2["t"] < ret["t"]:
                ret = ret2
                break
        return ret


class Cube:
    def __init__(self, trans_mat, bsdf):
        default_vertices = np.array([
            [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
            [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5]
        ])
        default_faces = np.array([
            [2, 1, 0],
            [0, 3, 2],
            [6, 5, 4],
            [4, 7, 6],
            [10, 9, 8],
            [8, 11, 10],
            [14, 13, 12],
            [12, 15, 14],
            [18, 17, 16],
            [16, 19, 18],
            [22, 21, 20],
            [20, 23, 22]
        ], np.uint)
        self.trans_mat = trans_mat
        self.mesh = trimesh.Trimesh(vertices=default_vertices,
                                    faces=default_faces,
                                    process=False)

        self.mesh.apply_transform(self.trans_mat)
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.bsdf = bsdf
        self.bounds = BBox(None, None)
        self.bounds.from_vertices(self.vertices)

    def visualize(self):
        self.mesh.show()

    def hit(self, ray):
        ret = {"origin": ray.position, "hit": False, "t": MAX_F,
               "position": np.array([0.0, 0.0, 0.0])}
        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            ret2 = triangle_ray_intersection([self.vertices[triangle[0]],
                                              self.vertices[triangle[1]],
                                              self.vertices[triangle[2]]], ray)
            if ret2["hit"] and ret2["t"] < ret["t"]:
                ret = ret2
        return ret


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
    mat = make_transformation_matrix(tr)
    a_quad = Quad(mat)
    a_quad.visualize()