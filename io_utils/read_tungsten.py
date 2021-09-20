import json
from core.scene import Scene
from mathematics.shapes import Quad, Cube
from mathematics.affine_transformation import make_transformation_matrix

PRIM_TYPES = {
    "quad": Quad,
    "cube": Cube,
}


def process_primitives(data):
    a_scene = Scene()
    for info in data['primitives']:
        trans_mat = make_transformation_matrix(info["transform"])
        if info["type"] not in PRIM_TYPES:
            print(f"[WARNING] {info['type']} not implemented")
            continue
        prim = PRIM_TYPES[info["type"]](trans_mat)
        a_scene.add_primitive(prim)
    a_scene.visualize()
    return a_scene


def read_file(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        process_primitives(data)
    return


if __name__ == '__main__':
    read_file("../media/cornell-box/scene.json")