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
    [1, 7.830270, [0.000000, 1.000000, 6.800000], [0.034281, 0.080880, -0.996134], [0.799974, -0.512694, 0.311747], [0.725000, 0.710000, 0.680000],
[-0.000000, 0.000000, 1.000000]],
    [1, 0.914496, [0.268427, 1.633309, -1.000000], [0.799974, -0.512694, 0.311747], [-0.944430, -0.165854, -0.283803], [0.140000, 0.450000, 0.091000],
[-1.000000, -0.000000, -0.000000]],
    [1, 1.004540, [1.000000, 1.164452, -0.714908], [-0.944430, -0.165854, -0.283803], [-0.539883, -0.658663, 0.524109], [0.725000, 0.710000, 0.680000],
[-0.000000, 0.000000, 1.000000]],
    [1, 0.766014, [0.051282, 0.997845, -1.000000], [-0.539883, -0.658663, 0.524109], [-0.039377, 0.827128, -0.560632], [0.725000, 0.710000, 0.680000],
[-0.328669, 0.000000, -0.944445]],
    [1, 0.716111, [-0.362276, 0.493300, -0.598526], [-0.039377, 0.827128, -0.560632], [0.688579, -0.690442, 0.221697], [0.725000, 0.710000, 0.680000],
[-0.000000, 0.000000, 1.000000]],
    [1, 1.572349, [-0.390474, 1.085616, -1.000000], [0.688579, -0.690442, 0.221697], [0.631949, 0.704407, -0.323189], [0.725000, 0.710000, 0.680000],
[-0.000000, 1.000000, -0.000000]],
    [1, 0.487045, [0.692212, 0.000000, -0.651414], [0.631949, 0.704407, -0.323189], [-0.074693, -0.669229, 0.739292], [0.140000, 0.450000, 0.091000],
[-1.000000, -0.000000, -0.000000]],
    [1, 0.512646, [1.000000, 0.343078, -0.808822], [-0.074693, -0.669229, 0.739292], [0.324842, 0.923928, -0.202077], [0.725000, 0.710000, 0.680000],
[-0.000000, 1.000000, -0.000000]],
    [1, 0.117876, [0.961709, -0.000000, -0.429826], [0.324842, 0.923928, -0.202077], [-0.052710, -0.073551, 0.995898], [0.140000, 0.450000, 0.091000],
[-1.000000, -0.000000, -0.000000]]
]

beta = np.ones((3,))
for d in data:
    _, t, ro, rd, wi, _, n = d

    ro = np.array(ro)
    rd = np.array(rd)
    n = np.array(n)
    wi = np.array(wi)
    res = a_scene.hit(ro, rd)
    print(beta, res["bsdf"], beta*res["bsdf"], beta*res["bsdf"]*255)
    beta *= res["bsdf"]

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
