import numpy as np
import trimesh
from .constants import MAX_F
import sys
from .intersection import triangle_ray_intersection, triangle_ray_intersection_grouping
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
        self.data = {}
        self.e1e2 = []
        self.triangles = []
        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            e1 = self.vertices[triangle[1]]-self.vertices[triangle[0]]
            e2 = self.vertices[triangle[2]]-self.vertices[triangle[0]]
            self.e1e2.extend([e1, e2])
            if i not in self.data:
                self.data[i] = [self.vertices[triangle[1]]-self.vertices[triangle[0]],
                                self.vertices[triangle[2]]-self.vertices[triangle[0]]]
            self.triangles.append([self.vertices[triangle[0]],
                                   self.vertices[triangle[1]],
                                   self.vertices[triangle[2]],
                                   self.data[i]])
        self.e1e2 = np.hstack(self.e1e2)
        self.cross_product = np.zeros((self.faces.shape[0]*2*3,), np.float64)
        self.cross_a = np.zeros_like(self.cross_product, np.float64)
        self.cross_b = np.zeros_like(self.cross_product, np.float64)
        self.s_holder = np.zeros((3,), np.float64)

    def visualize(self):
        self.mesh.show()

    def hit_faster(self, ray):
        results = triangle_ray_intersection_grouping(ray, self.triangles, self.e1e2,
                                                     self.cross_product, self.cross_a, self.cross_b,
                                                     self.s_holder)
        hit_results = [du for du in results if du["hit"]]
        if len(hit_results) > 0:
            ret = min(hit_results, key=lambda du: du["t"])
            ret["bsdf"] = self.bsdf
            return ret
        else:
            return {"hit": False}

    def hit(self, ray):
        return self.hit_faster(ray)


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
        self.data = {}
        self.e1e2 = []
        self.triangles = []
        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            e1 = self.vertices[triangle[1]]-self.vertices[triangle[0]]
            e2 = self.vertices[triangle[2]]-self.vertices[triangle[0]]
            self.e1e2.extend([e1, e2])
            if i not in self.data:
                self.data[i] = [self.vertices[triangle[1]]-self.vertices[triangle[0]],
                                self.vertices[triangle[2]]-self.vertices[triangle[0]]]
            self.triangles.append([self.vertices[triangle[0]],
                                   self.vertices[triangle[1]],
                                   self.vertices[triangle[2]],
                                   self.data[i]])
        self.e1e2 = np.hstack(self.e1e2)
        self.cross_product = np.zeros((self.faces.shape[0]*2*3,), np.float64)
        self.cross_a = np.zeros_like(self.cross_product, np.float64)
        self.cross_b = np.zeros_like(self.cross_product, np.float64)
        self.s_holder = np.zeros((3,), np.float64)

    def visualize(self):
        self.mesh.show()

    def hit_faster(self, ray):
        results = triangle_ray_intersection_grouping(ray, self.triangles, self.e1e2,
                                                     self.cross_product, self.cross_a,
                                                     self.cross_b, self.s_holder)
        hit_results = [du for du in results if du["hit"]]
        if len(hit_results) > 0:
            ret = min(hit_results, key=lambda du: du["t"])
            ret["bsdf"] = self.bsdf
            return ret
        else:
            return {"hit": False}

    def hit(self, ray):
        return self.hit_faster(ray)


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