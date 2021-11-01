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
data = [
    [1, 0, 7.828033, [0.000000, 1.000000, 6.800000], [-0.013691, 0.083438, -0.996419], [0.332904, -0.609201, 0.719757],
[-0.000000, 0.000000, 1.000000]],
    [1, 1, 1.728753, [-0.107174, 1.653158, -1.000000], [0.332904, -0.609201, 0.719757], [-0.901746, 0.432260, -0.002206],
[-0.000000, 1.000000, 0.000000]],
    [1, 2, 1.628323, [0.468334, 0.600000, 0.244282], [-0.901746, 0.432260, -0.002206], [0.432009, -0.866822, 0.248971],
[1.000000, -0.000000, 0.000000]],
    [1, 3, 1.504182, [-1.000000, 1.303859, 0.240690], [0.432009, -0.866822, 0.248971], [-0.936970, 0.111620, 0.331101],
[-0.000000, 1.000000, -0.000000]],
    [1, 4, 0.693534, [-0.350179, 0.000000, 0.615189], [-0.936970, 0.111620, 0.331101], [0.936005, 0.341610, -0.084832],
[1.000000, -0.000000, 0.000000]],
    [1, 5, 1.513839, [-1.000000, 0.077413, 0.844819], [0.936005, 0.341610, -0.084832], [-0.089066, -0.735613, 0.671521],
[-0.286357, -0.000000, 0.958123]]
]

for d in data:
    _, _, t, ro, rd, wi, n = d

    ro = np.array(ro)
    rd = np.array(rd)
    n = np.array(n)
    wi = np.array(wi)

    ray_logger.add_line(ro, ro+t*rd, color=[0, 0, 0])
    ray_logger.add_line(ro+t*rd, (ro+t*rd)+0.5*n)
    ray_logger.add_line(ro+t*rd, (ro+t*rd)+0.5*wi, color=[0, 0, 1])

# test intersection with specified ray
# ro = np.array([0.000000, 1.000000, 6.800000])
# rd = np.array([0.157926, 0.106794, -0.981659])
#
# tracks = []
# for bound in range(100):
#     res = a_scene.hit(ro, rd)
#     print(bound, res["t"], list(ro), list(rd))
#     if bound == 2:
#         break
#     tracks.append([list(ro), list(rd)])
#
#     if not res["hit"]:
#         break
#
#     pos = res["position"]
#     pos2 = ro + res["t"] * rd
#     ray_logger.add_line(ro, pos2, color=[0, 0, 0])
#
#     ray_logger.add_line(pos, pos + 0.5 * res["normal"])
#     ray_logger.add_line(pos, pos + 0.5 * res["wi"], color=[0, 1, 0])
#
#     ro = pos
#     rd = res["wi"]

# test normal side
# light_point = np.array([-0.122904, 1.980000, 0.066081])
# light_normal = np.array([0, -1, 0])
# ray_logger.add_line(light_point, light_normal * 0.5 + light_point, color=[0.5, 0.5, 0])
#
#
# def prepare_test(ro_, rd_):
#     res_ = a_scene.hit(ro_, rd_)
#     pos = res_["position"]
#     pos2 = ro + res_["t"] * rd
#     ray_logger.add_line(ro_, pos2, color=[0, 0, 0])
#     to_light = light_point-pos2
#     if np.dot(light_normal, -to_light) > 0:
#         ray_logger.add_line(pos2, light_point, color=[0, 1, 0])
#     else:
#         ray_logger.add_line(pos2, light_point, color=[1, 0, 0])
#
#     ray_logger.add_line(pos, pos + 0.5 * res_["normal"])
#     ray_logger.add_line(pos, pos + 0.5 * res_["wi"], color=[0, 1, 0])
#
#
# ro = np.array([0.000000, 1.000000, 6.800000])
# rd = np.array([0.8,  0.,  0]) - ro
# prepare_test(ro, rd)
# ro = np.array([0.000000, 1.000000, 6.800000])
# rd = np.array([-0.5,  2,  0]) - ro
# prepare_test(ro, rd)

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)

a_scene.visualize_o3d(line_set)
