import numpy as np
import trimesh
import random
from math import sqrt
from .samplers_debug import cosine_sample_hemisphere
from .intersection import triangle_ray_intersection_grouping
from .bbox import BBox
from .vec3 import normalize_vector


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
        normal_vector = np.array([0, 1, 0], np.float64)
        self.trans_mat = trans_mat
        self.normal_vectors = np.tile(normal_vector, (default_faces.shape[0], 1))
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
        self.triangles = []
        self.first_vertices = []
        self.e2 = []
        self.e1 = []
        self.normals = None

        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            e1 = self.vertices[triangle[1]]-self.vertices[triangle[0]]
            e2 = self.vertices[triangle[2]]-self.vertices[triangle[0]]
            self.normal_vectors[i] = -normalize_vector(np.cross(e1, e2))
            if i not in self.data:
                self.data[i] = [self.vertices[triangle[1]]-self.vertices[triangle[0]],
                                self.vertices[triangle[2]]-self.vertices[triangle[0]]]
            self.triangles.append([self.vertices[triangle[0]],
                                   self.vertices[triangle[1]],
                                   self.vertices[triangle[2]],
                                   self.data[i]])
            self.first_vertices.append(self.vertices[triangle[0]])
            self.e2.append(e2)
            self.e1.append(e1)

        self.s_array = np.zeros((self.faces.shape[0]*3,), np.float64)
        self.q_array = np.zeros((self.faces.shape[0]*3,), np.float64)
        self.r_array = np.zeros((self.faces.shape[0]*3,), np.float64)
        self.first_vertices = np.hstack(self.first_vertices)
        self.e2 = np.hstack(self.e2)
        self.e1 = np.hstack(self.e1)
        self.a_array = np.zeros((self.faces.shape[0],), np.float64)
        self.e2r_array = np.zeros((self.faces.shape[0],), np.float64)
        self.sq_array = np.zeros((self.faces.shape[0],), np.float64)
        self.rdr_array = np.zeros((self.faces.shape[0],), np.float64)
        self.res_array = np.zeros((self.faces.shape[0]*2,), np.float64)
        for i in range(self.faces.shape[0]):
            self.res_array[i*2] = -1.0

    def sample_a_point(self):
        face_id = random.randint(0, self.faces.shape[0]-1)
        u = sqrt(random.uniform(0, 1))
        v = random.uniform(0, 1)
        a = u*(1-v)
        b = u*v
        v0, v1, v2 = self.faces[face_id]
        return a*self.vertices[v0] + b*self.vertices[v1] + (1.0-a-b)*self.vertices[v2]

    def visualize(self):
        self.mesh.show()

    def hit_faster(self, ray):
        hit_results = triangle_ray_intersection_grouping(ray, len(self.triangles),
                                                         self.s_array, self.q_array, self.r_array,
                                                         self.first_vertices, self.e1, self.e2, self.a_array,
                                                         self.e2r_array, self.sq_array, self.rdr_array, self.res_array)
        if len(hit_results) > 0:
            ret, tri_ind = min(hit_results, key=lambda du: du[0]["t"])
            ret["bsdf"] = self.bsdf

            if np.dot(self.normal_vectors[tri_ind], -ray.direction) < 0.0:
                ret["normal"] = -self.normal_vectors[tri_ind]
            else:
                ret["normal"] = self.normal_vectors[tri_ind]

            direction = cosine_sample_hemisphere(ret["normal"])
            ret["wi"] = direction

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

        self.normal_vectors = np.array([
            [0, -1, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, 1],
            [0, 0, 1],
            [-1, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ], np.float64)
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
        self.triangles = []
        self.e2 = []
        self.e1 = []
        self.first_vertices = []
        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            e1 = self.vertices[triangle[1]]-self.vertices[triangle[0]]
            e2 = self.vertices[triangle[2]]-self.vertices[triangle[0]]
            self.normal_vectors[i] = normalize_vector(np.cross(e1, e2))
            if i not in self.data:
                self.data[i] = [self.vertices[triangle[1]]-self.vertices[triangle[0]],
                                self.vertices[triangle[2]]-self.vertices[triangle[0]]]
            self.triangles.append([self.vertices[triangle[0]],
                                   self.vertices[triangle[1]],
                                   self.vertices[triangle[2]],
                                   self.data[i]])
            self.first_vertices.append(self.vertices[triangle[0]])
            self.e2.append(e2)
            self.e1.append(e1)
        self.s_array = np.zeros((self.faces.shape[0]*3,), np.float64)
        self.q_array = np.zeros((self.faces.shape[0]*3,), np.float64)
        self.r_array = np.zeros((self.faces.shape[0]*3,), np.float64)
        self.e2 = np.hstack(self.e2)
        self.e1 = np.hstack(self.e1)
        self.first_vertices = np.hstack(self.first_vertices)
        self.a_array = np.zeros((self.faces.shape[0],), np.float64)
        self.e2r_array = np.zeros((self.faces.shape[0],), np.float64)
        self.sq_array = np.zeros((self.faces.shape[0],), np.float64)
        self.rdr_array = np.zeros((self.faces.shape[0],), np.float64)
        self.res_array = np.zeros((self.faces.shape[0]*2,), np.float64)
        for i in range(self.faces.shape[0]):
            self.res_array[i*2] = -1.0

    def visualize(self):
        self.mesh.show()

    def hit_faster(self, ray):
        hit_results = triangle_ray_intersection_grouping(ray, len(self.triangles),
                                                         self.s_array, self.q_array, self.r_array,
                                                         self.first_vertices, self.e1, self.e2, self.a_array,
                                                         self.e2r_array, self.sq_array, self.rdr_array, self.res_array)
        if len(hit_results) > 0:
            ret, tri_ind = min(hit_results, key=lambda du: du[0]["t"])
            ret["bsdf"] = self.bsdf
            if np.dot(self.normal_vectors[tri_ind], -ray.direction) < 0.0:
                ret["normal"] = -self.normal_vectors[tri_ind]
            else:
                ret["normal"] = self.normal_vectors[tri_ind]

            direction = cosine_sample_hemisphere(ret["normal"])
            ret["wi"] = direction

            return ret
        else:
            return {"hit": False}

    def hit(self, ray):
        return self.hit_faster(ray)
