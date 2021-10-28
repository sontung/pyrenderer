import json
from core.scene import Scene
from core.bsdf import BSDF
from core.camera import Camera
from mathematics.shapes import Quad, Cube

from mathematics.affine_transformation import make_transformation_matrix

PRIM_TYPES = {
    "quad": Quad,
    "cube": Cube,
}


def process_primitives(data):
    a_scene = Scene()
    name2bsdf = {}

    # read camera
    a_camera = Camera(data["camera"]["transform"]["position"],
                      data["camera"]["transform"]["look_at"],
                      data["camera"]["transform"]["up"],
                      data["camera"]["resolution"],
                      fov=data["camera"]["fov"])

    # read bsdfs
    for info_bsdf in data["bsdfs"]:
        name2bsdf[info_bsdf["name"]] = BSDF(info_bsdf).get_distribution()

    # read primitives
    for info in data['primitives']:
        trans_mat = make_transformation_matrix(info["transform"])
        if info["type"] not in PRIM_TYPES:
            print(f"[WARNING] {info['type']} not implemented")
            continue
        prim = PRIM_TYPES[info["type"]](trans_mat, name2bsdf[info["bsdf"]])
        a_scene.add_primitive(prim)
    # a_scene.visualize()
    # a_scene.build_bvh_tree()
    return a_scene, a_camera


def read_file(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return process_primitives(data)


if __name__ == '__main__':
    read_file("../media/cornell-box/scene.json")