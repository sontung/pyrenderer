from io_utils.read_utils_debug import read_scene
from debug.ray_logger import RayLogger
from mathematics.samplers_debug import cosine_sample_hemisphere
import numpy as np
import open3d as o3d

a_scene = read_scene("media/cornell-box/scene.json")

ray_logger = RayLogger()

data = [
    [1, 0, 6.759408, [0.000000, 1.000000, 6.800000], [-0.061724, 0.001867, -0.998092], [-0.214061, 0.541850, 0.812758],
[-0.328669, 0.000000, -0.944445]],
]

for d in data:
    _, _, t, ro, rd, wi, n = d

    ro = np.array(ro)
    rd = np.array(rd)
    n = np.array(n)
    wi = np.array(wi)

    # ray_logger.add_line(ro, ro+t*rd)
    ray_logger.add_line(ro+t*rd, (ro+t*rd)-0.5*n)
    ray_logger.add_line(ro+t*rd, (ro+t*rd)+0.5*wi, color=[0, 0, 1])

    ray_logger.add_line(ro+t*rd, (ro+t*rd)-0.5*wi, color=[0, 1, 0])




line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)
a_scene.visualize_o3d(line_set)
