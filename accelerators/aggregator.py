import numpy as np
from mathematics.intersection import triangle_ray_intersection_grouping


class Aggregator:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.normals = None
        self.triangles = []
        self.data = {}
        self.triangle2prim_info = {}
        self.first_vertices = []
        self.e2 = []
        self.e1 = []
        self.q_array = []
        self.r_array = []
        self.s_array = []
        self.a_array = []
        self.e2r_array = []
        self.sq_array = []
        self.rdr_array = []
        self.res_array = []

    def update(self):
        self.triangles = []
        self.first_vertices = []
        self.e2 = []
        self.e1 = []
        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            e1 = self.vertices[triangle[1]] - self.vertices[triangle[0]]
            e2 = self.vertices[triangle[2]] - self.vertices[triangle[0]]
            if i not in self.data:
                self.data[i] = [self.vertices[triangle[1]] - self.vertices[triangle[0]],
                                self.vertices[triangle[2]] - self.vertices[triangle[0]]]
            self.triangles.append([self.vertices[triangle[0]],
                                   self.vertices[triangle[1]],
                                   self.vertices[triangle[2]]])
            self.first_vertices.append(self.vertices[triangle[0]])
            self.e2.append(e2)
            self.e1.append(e1)
        self.q_array = np.zeros((self.faces.shape[0]*3,), np.float32)
        self.r_array = np.zeros((self.faces.shape[0]*3,), np.float32)
        self.s_array = np.zeros((self.faces.shape[0]*3,), np.float32)
        self.first_vertices = np.hstack(self.first_vertices)
        self.e2 = np.hstack(self.e2)
        self.e1 = np.hstack(self.e1)
        self.a_array = np.zeros((self.faces.shape[0],), np.float32)
        self.e2r_array = np.zeros((self.faces.shape[0],), np.float32)
        self.sq_array = np.zeros((self.faces.shape[0],), np.float32)
        self.rdr_array = np.zeros((self.faces.shape[0],), np.float32)
        self.res_array = np.zeros((self.faces.shape[0]*2,), np.float32)
        for i in range(self.faces.shape[0]):
            self.res_array[i*2] = -1.0

    def push(self, primitive):
        if self.vertices is None:
            self.vertices = primitive.vertices
            self.normals = primitive.normal_vectors
            self.faces = primitive.faces
            for i in range(self.faces.shape[0]):
                self.triangle2prim_info[i] = [primitive.bsdf]
        else:
            increment = self.vertices.shape[0]
            old_faces_size = self.faces.shape[0]
            self.vertices = np.vstack([self.vertices, primitive.vertices])
            self.normals = np.vstack([self.normals, primitive.normal_vectors])
            self.faces = np.vstack([self.faces, primitive.faces+increment])
            for i in range(old_faces_size, self.faces.shape[0]):
                self.triangle2prim_info[i] = [primitive.bsdf]
        self.update()

    def hit(self, ray):
        hit_results = triangle_ray_intersection_grouping(ray, len(self.triangles),
                                                         self.s_array, self.q_array, self.r_array,
                                                         self.first_vertices, self.e1, self.e2, self.a_array,
                                                         self.e2r_array, self.sq_array, self.rdr_array, self.res_array)
        if len(hit_results) > 0:
            ret, idx = min(hit_results, key=lambda du: du[0]["t"])
            ret["bsdf"] = self.triangle2prim_info[idx][0]
            ret["normal"] = self.normals[idx]
            return ret
        else:
            return {"hit": False}

