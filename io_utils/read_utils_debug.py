import json
import numpy as np
import open3d as o3d
from core.ray import Ray
from mathematics.shapes2 import Quad as Quad_debug
from mathematics.shapes2 import Cube as Cube_debug
from mathematics.affine_transformation import make_transformation_matrix


class Scene:
    def __init__(self):
        self.primitives = []
        self.vertices = None
        self.faces = None
        self.mesh = None
        self.bvh_compatible_prims = []
        self.bvh_not_compatible_prims = []
        self.lights = []
        self.primitives2 = []

    def add_primitive(self, prim):
        self.primitives.append(prim)
        try:
            prim = prim.normal_shape()
            self.primitives2.append(prim)
        except AttributeError:
            pass
        if self.vertices is None:
            self.vertices = prim.vertices
            self.faces = prim.faces
        else:
            increment = self.vertices.shape[0]
            self.vertices = np.vstack([self.vertices, prim.vertices])
            self.faces = np.vstack([self.faces, prim.faces+increment])

    def hit(self, ro, rd):
        ray = Ray(ro, rd)
        ret = {"origin": ray.position, "hit": False, "t": 999.9,
               "position": np.array([0.0, 0.0, 0.0])}
        for prim in self.primitives:
            ret2 = prim.hit(ray)
            if ret2["hit"] and ret2["t"] < ret["t"]:
                ret = ret2
        return ret

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


PRIM_TYPES_debug = {
    "quad": Quad_debug,
    "cube": Cube_debug,
}


def read_scene(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        a_scene = Scene()

        # read primitives
        for info in data['primitives']:
            trans_mat = make_transformation_matrix(info["transform"])
            if info["type"] not in PRIM_TYPES_debug:
                print(f"[WARNING] {info['type']} not implemented")
                continue
            prim = PRIM_TYPES_debug[info["type"]](trans_mat, None)
            a_scene.add_primitive(prim)
    return a_scene
