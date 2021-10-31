from io_utils.read_utils_debug import read_scene
from debug.ray_logger import RayLogger
from mathematics.samplers_debug import cosine_sample_hemisphere2, cosine_sample_hemisphere
import numpy as np
import sys
import open3d as o3d


np.random.seed(2)
a_scene = read_scene("media/cornell-box/scene.json")

ray_logger = RayLogger()

# test cosine sampling
# ro = np.array([0.0, 0.0, 0.0])
# normal = np.array([0, 1, 0])
# for _ in range(40):
#     wi2 = cosine_sample_hemisphere(normal)
#     ray_logger.add_line(ro, ro + 0.5 * wi2, color=[1, 0, 0])
#
# ray_logger.add_line(ro, ro + 0.5 * normal, color=[0, 0, 1])
#
# line_set = o3d.geometry.LineSet()
# line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
# line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
# line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)
#
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(line_set)
#
# while True:
#     vis.poll_events()
#     vis.update_renderer()
# sys.exit()

# recreate previous tracing
# data = [
#     [1, 0, 6.332074, [0.000000, 1.000000, 6.800000], [0.157926, 0.106794, -0.981659], [-0.808240, -0.580431, 0.099233],
# [-1.000000, -0.000000, -0.000000]],
#     [1, 1, 2.474511, [1.000000, 1.676225, 0.584062], [-0.808240, -0.580431, 0.099233], [0.953661, -0.288658, -0.084895],
# [1.000000, -0.000000, 0.000000]],
#     [1, 2, 0.831235, [-1.000000, 0.239942, 0.829615], [0.953661, -0.288658, -0.084895], [0.167221, 0.568295, 0.805654],
# [-0.000000, 1.000000, -0.000000]]
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
#     ray_logger.add_line(ro, ro+t*rd, color=[0, 0, 0])
#     ray_logger.add_line(ro+t*rd, (ro+t*rd)+0.5*n)
#     ray_logger.add_line(ro+t*rd, (ro+t*rd)+0.5*wi, color=[0, 0, 1])

# test intersection with specified ray
ro = np.array([0.000000, 1.000000, 6.800000])
rd = np.array([0.157926, 0.106794, -0.981659])

tracks = []
for bound in range(100):
    res = a_scene.hit(ro, rd)
    print(bound, res["t"], list(ro), list(rd))
    if bound == 2:
        break
    tracks.append([list(ro), list(rd)])

    if not res["hit"]:
        break

    pos = res["position"]
    pos2 = ro + res["t"] * rd
    ray_logger.add_line(ro, pos2, color=[0, 0, 0])

    ray_logger.add_line(pos, pos + 0.5 * res["normal"])
    ray_logger.add_line(pos, pos + 0.5 * res["wi"], color=[0, 1, 0])

    ro = pos
    rd = res["wi"]


line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)

a_scene.visualize_o3d(line_set)
