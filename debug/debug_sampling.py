from io_utils.read_tungsten import read_file
from debug.ray_logger import RayLogger
from mathematics.samplers import cosine_sample_hemisphere
from mathematics.intersection_taichi import World
from taichi_glsl.randgen import rand
from taichi_glsl.vector import vec2
import taichi as ti
import numpy as np
import open3d as o3d

ti.init(arch=ti.gpu, debug=True)
a_scene, a_camera = read_file("media/cornell-box/scene.json")
world = World()
for p in a_scene.primitives:
    world.add(p)
world.commit()

direction_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(3, 40))
pos_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(3, 1))
normal_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(3, 1))


@ti.kernel
def main():
    for j in range(3):
        u, n = world.sample_a_point()
        pos_field[j, 0] = u
        normal_field[j, 0] = u
        for i in range(40):
            direction = cosine_sample_hemisphere(vec2(rand(), rand()))
            direction_field[j, i] = direction


def run(a_scene, a_camera):
    ray_logger = RayLogger()

    # for i in range(len(data)):
    #     ro, rd, t, nd, no = data[i]
    #     ray_logger.add_line(np.array(ro), np.array(rd)*t+np.array(ro))
    #     ray_logger.add_line(np.array(no), np.array(nd)*0.5+np.array(no), color=[0, 1, 0])


    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
    line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
    line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)
    a_scene.visualize_o3d(line_set)
