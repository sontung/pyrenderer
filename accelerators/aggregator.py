import numpy as np
from mathematics.intersection import triangle_ray_intersection_grouping


class Aggregator:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.e1e2 = []
        self.triangles = []
        self.data = {}
        self.cross_product = None
        self.cross_a = None
        self.cross_b = None
        self.s_holder = np.zeros((3,), np.float64)
        self.triangle2prim_info = {}

    def update(self):
        self.e1e2 = []
        self.triangles = []
        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            e1 = self.vertices[triangle[1]] - self.vertices[triangle[0]]
            e2 = self.vertices[triangle[2]] - self.vertices[triangle[0]]
            self.e1e2.extend([e1, e2])
            if i not in self.data:
                self.data[i] = [self.vertices[triangle[1]] - self.vertices[triangle[0]],
                                self.vertices[triangle[2]] - self.vertices[triangle[0]]]
            self.triangles.append([self.vertices[triangle[0]],
                                   self.vertices[triangle[1]],
                                   self.vertices[triangle[2]],
                                   self.data[i]])
        self.e1e2 = np.hstack(self.e1e2)
        self.cross_product = np.zeros((self.faces.shape[0] * 2 * 3,), np.float64)
        self.cross_a = np.zeros_like(self.cross_product, np.float64)
        self.cross_b = np.zeros_like(self.cross_product, np.float64)
        self.s_holder = np.zeros((3,), np.float64)

    def push(self, primitive):
        if self.vertices is None:
            self.vertices = primitive.vertices
            self.faces = primitive.faces
            for i in range(self.faces.shape[0]):
                self.triangle2prim_info[i] = [primitive.bsdf]
        else:
            increment = self.vertices.shape[0]
            old_faces_size = self.faces.shape[0]
            self.vertices = np.vstack([self.vertices, primitive.vertices])
            self.faces = np.vstack([self.faces, primitive.faces+increment])
            for i in range(old_faces_size, self.faces.shape[0]):
                self.triangle2prim_info[i] = [primitive.bsdf]
        self.update()

    def hit(self, ray):
        results = triangle_ray_intersection_grouping(ray, self.triangles, self.e1e2,
                                                     self.cross_product, self.cross_a, self.cross_b,
                                                     self.s_holder)
        hit_results = [(du, idx) for idx, du in enumerate(results) if du["hit"]]
        if len(hit_results) > 0:
            ret, idx = min(hit_results, key=lambda du: du[0]["t"])
            ret["bsdf"] = self.triangle2prim_info[idx][0]
            return ret
        else:
            return {"hit": False}