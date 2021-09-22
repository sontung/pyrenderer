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

    def hit(self, ray):
        ret = {"origin": ray.position, "hit": False, "t": 0.0,
               "position": np.array([0.0, 0.0, 0.0])}
        for prim in self.primitives:
            ret2 = prim.hit(ray)
            if ret2["hit"] and ret2["t"] < ret["t"]:
                ret = ret2
        return ret

    def visualize(self):
        self.mesh = trimesh.Trimesh(vertices=self.vertices,
                                    faces=self.faces,
                                    process=False)
        self.mesh.show()
