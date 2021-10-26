from io_utils.read_tungsten import read_file
from debug.ray_logger import RayLogger
import taichi as ti
import numpy as np
import open3d as o3d

ti.init()
data = [
    ([0.000000, 1.000000, 6.800000], [0.134840, 0.132964, -0.981906], 7.416195, [-1.000000, -0.000000, -0.000000], [1.000000, 1.986086, -0.482004]),
    # ([1.000000, 1.986086, -0.482004], [0.427235, 0.846925, 0.316525], 1.898438, [0.000000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000]),
    # ([1.000000, 1.131270, 0.848348], [-0.282163, 0.662193, 0.694179], 0.5)
    # ([0.335731, 0.600000, 0.528806], [0.413983, 0.155196, 0.896957], 10)
]

a_scene, a_camera = read_file("media/cornell-box/scene.json")

x_dim, y_dim = a_camera.get_resolution()
image = np.zeros((x_dim, y_dim, 3), dtype=np.float32)
ray_logger = RayLogger()


s1, s2 = [[-0.846925, 0.316525, -0.427235], [-0.316525, -0.427235, 0.846925]]

from mathematics.mat4 import rotate_vector, rotate_to
m1 = rotate_to(np.array([-1.000000, -0.000000, -0.000000]))
s3 = rotate_vector(m1, np.array(s1))
print(s3)

t0, t1, ro, rd = 1.353912, 1.353912, [1.000000, 1.170249, -0.548879], [-0.636713, 0.598082, 0.486718]
ro = np.array(ro)
rd = np.array(rd)

ray_logger.add_line(ro, ro+rd*t0)
ray_logger.add_line(ro, ro+rd*t1, color=[0, 1, 0])

# for i in range(len(data)):
#     ro, rd, t, nd, no = data[i]
#     ray_logger.add_line(np.array(ro), np.array(rd)*t+np.array(ro))
#     ray_logger.add_line(np.array(no), np.array(nd)*0.5+np.array(no), color=[0, 1, 0])


line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)
a_scene.visualize_o3d(line_set)
