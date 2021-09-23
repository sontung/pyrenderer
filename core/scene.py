import trimesh
import numpy as np
from accelerators.bvh import BVH


class Scene:
    def __init__(self):
        self.primitives = []
        self.vertices = None
        self.faces = None
        self.mesh = None
        self.tree = BVH()
        self.bvh_compatible_prims = []
        self.bvh_not_compatible_prims = []

    def add_primitive(self, prim):
        self.primitives.append(prim)
        if self.vertices is None:
            self.vertices = prim.vertices
            self.faces = prim.faces
        else:
            increment = self.vertices.shape[0]
            self.vertices = np.vstack([self.vertices, prim.vertices])
            self.faces = np.vstack([self.faces, prim.faces+increment])

    def build_bvh_tree(self):
        print("building BVH tree")

        for prim in self.primitives:
            if prim.bounds.is_empty():
                self.bvh_not_compatible_prims.append(prim)
            else:
                self.bvh_compatible_prims.append(prim)

        self.tree.build(self.bvh_compatible_prims)

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
