from io_utils.read_tungsten import read_file
from debug.ray_logger import RayLogger
import taichi as ti
import numpy as np
import open3d as o3d

ti.init()
data = [
    ([0.000000, 1.000000, 6.800000], [0.073997, 0.166125, -0.983324], 6.019572),
    ([0.445433, 2.000000, 0.880808], [0.537809, -0.842479, -0.031479], 1.031160),
    ([1.000000, 1.131270, 0.848348], [-0.282163, 0.662193, 0.694179], 0.5)
    # ([0.335731, 0.600000, 0.528806], [0.413983, 0.155196, 0.896957], 10)
]

a_scene, a_camera = read_file("media/cornell-box/scene.json")

x_dim, y_dim = a_camera.get_resolution()
image = np.zeros((x_dim, y_dim, 3), dtype=np.float32)
ray_logger = RayLogger()

for i in range(len(data)):
    ro, rd, t = data[i]
    ray_logger.add_line(np.array(ro), np.array(rd)*t+np.array(ro))

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)
a_scene.visualize_o3d(line_set)
