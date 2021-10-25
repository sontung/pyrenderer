import numpy as np
import trimesh
import random
import taichi as ti
from math import sqrt
from .constants import MAX_F
from .intersection_taichi import ray_triangle_hit, ray_triangle_hit2
from .bbox import BBox
from .vec3 import normalize_vector
from .shapes2 import Quad as Quad2
from .shapes2 import Cube as Cube2
from taichi_glsl.randgen import randInt, rand


@ti.data_oriented
class Quad:
    def __init__(self, trans_mat, bsdf):
        self.id = -1
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
        normal_vector = np.array([0, 1, 0], np.float32)
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
        self.center = self.bounds.center()

        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            e1 = self.vertices[triangle[1]]-self.vertices[triangle[0]]
            e2 = self.vertices[triangle[2]]-self.vertices[triangle[0]]
            self.normal_vectors[i] = -normalize_vector(np.cross(e1, e2))

        self.vertices_ti = ti.Vector.field(n=3, dtype=ti.f32, shape=(default_vertices.shape[0], 1))
        self.faces_ti = ti.Vector.field(n=3, dtype=ti.uint32, shape=(default_faces.shape[0], 1))
        self.normals_ti = ti.Vector.field(n=3, dtype=ti.f32, shape=(default_faces.shape[0], 1))
        for a in range(default_vertices.shape[0]):
            self.vertices_ti[a, 0] = self.vertices[a]
        for a in range(default_faces.shape[0]):
            self.faces_ti[a, 0] = self.faces[a]
        for a in range(default_faces.shape[0]):
            self.normals_ti[a, 0] = self.normal_vectors[a]

    def normal_shape(self):
        return Quad2(self.trans_mat, self.bsdf)

    @ti.func
    def sample_a_point(self):
        face_id = randInt(0, self.faces.shape[0]-1)
        u = ti.sqrt(rand())
        v = rand()
        a = u*(1-v)
        b = u*v
        v0, v1, v2 = self.faces_ti[face_id, 0]
        return a*self.vertices_ti[v0, 0] + b*self.vertices_ti[v1, 0] + (1.0-a-b)*self.vertices_ti[v2, 0]

    def visualize(self):
        self.mesh.show()

    @ti.func
    def hit(self, ro, rd, t0, t1):
        t_min = MAX_F
        hit_anything = 0
        face_id = 0
        for i in range(self.faces.shape[0]):
            face = self.faces_ti[i, 0]
            hit, t = ray_triangle_hit(self.vertices_ti[face[0], 0], self.vertices_ti[face[1], 0],
                                      self.vertices_ti[face[2], 0], ro, rd, t0, t1)

            if hit > 0 and t < t_min:
                t1 = t
                t_min = t
                face_id = i
                hit_anything = 1

        next_rd, attenuation = self.bsdf.scatter()
        emit, sided = self.bsdf.emitting_light, self.bsdf.sided
        return hit_anything, t_min, self.normals_ti[face_id, 0], next_rd, attenuation, emit, sided

    @property
    def bounding_box(self):
        return self.bounds.min_coord, self.bounds.max_coord


@ti.data_oriented
class Cube:
    def __init__(self, trans_mat, bsdf):
        self.id = -1
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
        ], np.float32)
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
        self.center = self.bounds.center()
        for i in range(self.faces.shape[0]):
            triangle = self.faces[i]
            e1 = self.vertices[triangle[1]]-self.vertices[triangle[0]]
            e2 = self.vertices[triangle[2]]-self.vertices[triangle[0]]
            self.normal_vectors[i] = normalize_vector(np.cross(e1, e2))

        self.vertices_ti = ti.Vector.field(n=3, dtype=ti.f32, shape=(default_vertices.shape[0], 1))
        self.faces_ti = ti.Vector.field(n=3, dtype=ti.uint32, shape=(default_faces.shape[0], 1))
        self.normals_ti = ti.Vector.field(n=3, dtype=ti.f32, shape=(default_faces.shape[0], 1))
        for a in range(default_vertices.shape[0]):
            self.vertices_ti[a, 0] = self.vertices[a]
        for a in range(default_faces.shape[0]):
            self.faces_ti[a, 0] = self.faces[a]
        for a in range(default_faces.shape[0]):
            self.normals_ti[a, 0] = self.normal_vectors[a]

    @ti.func
    def sample_a_point(self):
        face_id = randInt(0, self.faces.shape[0]-1)
        u = ti.sqrt(rand())
        v = rand()
        a = u*(1-v)
        b = u*v
        v0, v1, v2 = self.faces_ti[face_id, 0]
        return a*self.vertices_ti[v0, 0] + b*self.vertices_ti[v1, 0] + (1.0-a-b)*self.vertices_ti[v2, 0]

    def normal_shape(self):
        return Cube2(self.trans_mat, self.bsdf)

    def visualize(self):
        self.mesh.show()

    @ti.func
    def hit(self, ro, rd, t0, t1):
        t_min = MAX_F
        hit_anything = 0
        face_id = 0
        for i in range(self.faces.shape[0]):
            face = self.faces_ti[i, 0]
            hit, t = ray_triangle_hit(self.vertices_ti[face[0], 0], self.vertices_ti[face[1], 0],
                                      self.vertices_ti[face[2], 0], ro, rd, t0, t1)
            if hit > 0 and t < t_min:
                t1 = t
                t_min = t
                face_id = i
                hit_anything = 1

        next_rd, attenuation = self.bsdf.scatter()
        emit, sided = self.bsdf.emitting_light, self.bsdf.sided
        return hit_anything, t_min, self.normals_ti[face_id, 0], next_rd, attenuation, emit, sided

    @property
    def bounding_box(self):
        return self.bounds.min_coord, self.bounds.max_coord
