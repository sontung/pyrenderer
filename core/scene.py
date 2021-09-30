import trimesh
import numpy as np
import open3d as o3d
from accelerators.bvh import BVH
from accelerators.aggregator import Aggregator
from mathematics.constants import MAX_F


class Scene:
    def __init__(self):
        self.primitives = []
        self.vertices = None
        self.faces = None
        self.mesh = None
        self.tree_small = BVH()
        self.bvh_compatible_prims = []
        self.bvh_not_compatible_prims = []
        self.aggregator = Aggregator()

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
        for prim in self.primitives:
            if prim.bounds.is_empty():
                self.aggregator.push(prim)
                self.bvh_not_compatible_prims.append(prim)
            else:
                self.bvh_compatible_prims.append(prim)
        self.tree_small.build(self.bvh_compatible_prims)
        self.primitives = self.bvh_not_compatible_prims
        self.primitives.extend([self.bvh_compatible_prims])

    def hit_faster(self, ray):
        ret = self.tree_small.hit(ray)
        ret2 = self.aggregator.hit(ray)
        if ret2["hit"] and ret2["t"] < ret["t"]:
            ret = ret2
        return ret

    def hit(self, ray):
        ret = {"origin": ray.position, "hit": False, "t": MAX_F,
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

    def visualize_o3d(self, lines):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.vertices),
                                         o3d.utility.Vector3iVector(self.faces))
        vis.add_geometry(mesh)
        vis.add_geometry(lines)

        while True:
            vis.poll_events()
            vis.update_renderer()

