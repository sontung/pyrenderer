import trimesh
import numpy as np


class Scene:
    def __init__(self):
        self.primitives = []
        self.vertices = None
        self.faces = None
        self.mesh = None

    def add_primitive(self, prim):
        self.primitives.append(prim)
        if self.vertices is None:
            self.vertices = prim.vertices
            self.faces = prim.faces
        else:
            increment = self.vertices.shape[0]
            self.vertices = np.vstack([self.vertices, prim.vertices])
            self.faces = np.vstack([self.faces, prim.faces+increment])

    def visualize(self):
        self.mesh = trimesh.Trimesh(vertices=self.vertices,
                                    faces=self.faces,
                                    process=False)
        print()
        self.mesh.show()
