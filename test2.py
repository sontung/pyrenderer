from io_utils.read_tungsten import read_file
from debug.ray_logger import RayLogger
from mathematics.samplers import cosine_sample_hemisphere
from mathematics.intersection_taichi import World
from mathematics.mat4_taichi import rotate_to, rotate_z_to, rotate_vector
from taichi_glsl.randgen import rand
from taichi_glsl.vector import vec2
import taichi as ti
import open3d as o3d

ti.init(arch=ti.gpu, debug=True)
a_scene, a_camera = read_file("media/cornell-box/scene.json")
world = World()
for p in a_scene.primitives:
    world.add(p)
world.commit()

direction_field2 = ti.Vector.field(n=3, dtype=ti.f32, shape=(3, 40))
direction_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(3, 40))
pos_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(3, 1))
normal_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(3, 1))


@ti.kernel
def main():
    for j in range(3):
        u, n = world.sample_a_point()
        pos_field[j, 0] = u
        normal_field[j, 0] = n
        for i in range(40):
            direction = cosine_sample_hemisphere(vec2(rand(), rand()))
            # r1, r2, r3, r4 = rotate_to(n)
            # direction_field[j, i] = rotate_vector(r1, r2, r3, direction)
            r1, r2, r3, r4 = rotate_z_to(n)
            direction_field2[j, i] = rotate_vector(r1, r2, r3, direction)

main()
ray_logger = RayLogger()

for i in range(3):
    pos = pos_field[i, 0].to_numpy()
    normal = normal_field[i, 0].to_numpy()
    ray_logger.add_line(pos, normal*0.3+pos, color=[0, 1, 0])
    for j in range(40):
        direction = direction_field[i, j].to_numpy()
        ray_logger.add_line(pos, direction * 0.3 + pos)
        direction = direction_field2[i, j].to_numpy()
        ray_logger.add_line(pos, direction * 0.3 + pos, color=[0.5, 0.5, 0.5])


line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(ray_logger.points)
line_set.lines = o3d.utility.Vector2iVector(ray_logger.lines)
line_set.colors = o3d.utility.Vector3dVector(ray_logger.colors)
a_scene.visualize_o3d(line_set)
