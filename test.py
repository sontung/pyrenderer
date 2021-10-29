from io_utils.read_utils_debug import read_scene
from debug.ray_logger import RayLogger
from mathematics.samplers_debug import cosine_sample_hemisphere2, cosine_sample_hemisphere
import numpy as np
import open3d as o3d

a_scene = read_scene("media/cornell-box/scene.json")

ray_logger = RayLogger()

# data = [
#     [1, 0, 6.759408, [0.000000, 1.000000, 6.800000], [-0.061724, 0.001867, -0.998092], [0.214061, 0.541850, -0.812758],
# [0.328669, -0.000000, 0.944445]],
#     [1, 1, 0.345816, [-0.417215, 1.012619, 0.053493], [0.214061, 0.541850, -0.812758], [0.165413, 0.396828, 0.902866],
# [-0.000000, -1.000000, -0.000000]]
# ]
#
# for d in data:
#     _, _, t, ro, rd, wi, n = d
#
#     ro = np.array(ro)
#     rd = np.array(rd)
#     n = np.array(n)
#     wi = np.array(wi)
#
#     # ray_logger.add_line(ro, ro+t*rd)
#     ray_logger.add_line(ro+t*rd, (ro+t*rd)+0.5*n)
#     # ray_logger.add_line(ro+t*rd, (ro+t*rd)+0.5*wi, color=[0, 0, 1])
#     ray_logger.add_line(ro+t*rd, (ro+t*rd)-0.5*wi, color=[0, 1, 0])


ro = np.array([0.000000, 1.000000, 6.800000])
rd = np.array([-0.061724, 0.001867, -0.998092])
res = a_scene.hit(ro, rd)
while True:
    print(res)
    if not res["hit"]:
        break
    pos = res["position"]
    ray_logger.add_line(pos, pos + 0.5 * res["normal"])
    ray_logger.add_line(pos, pos + 0.5 * res["wi"], color=[0, 1, 0])

    ro = pos
    rd = res["wi"]
    res = a_scene.hit(ro, rd)

# ro = np.array([0.0, 0.0, 0.0])
# normal = np.array([0, 0, 1])
# for _ in range(40):
#     wi2, wi = cosine_sample_hemisphere(normal)
#     ray_logger.add_line(ro, ro + 0.5 * wi2, color=[1, 0, 0])
#
# ray_logger.add_line(ro, ro + 0.5 * normal, color=[0, 0, 1])


line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(line_set)
#
# while True:
#     vis.poll_events()
#     vis.update_renderer()

a_scene.visualize_o3d(line_set)
