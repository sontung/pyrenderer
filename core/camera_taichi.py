import taichi as ti
from taichi_glsl.vector import normalize, dot
from taichi_glsl.randgen import rand
from math import tan, radians
from mathematics.vec3_taichi import *
import math


@ti.data_oriented
class Camera:
    def __init__(self, vfrom, at, up, fov, aspect_ratio, aperture, focus_dist, iview_mat):
        theta = math.radians(fov)
        h = math.tan(theta / 2.0)
        viewport_height = 2.0 * h
        viewport_width = viewport_height * aspect_ratio

        w = (vfrom - at).normalized()
        u = up.cross(w).normalized()
        v = w.cross(u)

        self.origin = vfrom
        self.horizontal = focus_dist * viewport_width * u
        self.vertical = focus_dist * viewport_height * v
        self.lower_left_corner = self.origin - (self.horizontal / 2.0) \
                                 - (self.vertical / 2.0) \
                                 - focus_dist * w
        self.lens_radius = aperture / 2.0
        self.iview_c1 = Vector4(iview_mat[0][0], iview_mat[0][1],
                                iview_mat[0][2], iview_mat[0][3])
        self.iview_c2 = Vector4(iview_mat[1][0], iview_mat[1][1],
                                iview_mat[1][2], iview_mat[1][3])
        self.iview_c3 = Vector4(iview_mat[2][0], iview_mat[2][1],
                                iview_mat[2][2], iview_mat[2][3])
        self.iview_c4 = Vector4(iview_mat[3][0], iview_mat[3][1],
                                iview_mat[3][2], iview_mat[3][3])
        self.sensor_height = tan(radians(fov) / 2) * focus_dist
        self.sensor_width = self.sensor_height * aspect_ratio
        self.sensor_dim = Vector4(self.sensor_width,
                                  self.sensor_height, focus_dist, aperture)

    @ti.func
    def get_ray(self, u, v):
        rd = self.lens_radius * random_in_unit_disk()
        offset = u * rd.x + v * rd.y
        return self.origin + offset, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin - offset

    @ti.func
    def gen_ray(self, u, v):
        ray_dir = Vector4((u-0.5) * self.sensor_dim[0] / 0.5,
                          (v-0.5) * self.sensor_dim[1] / 0.5,
                          -self.sensor_dim[2], 1.0)

        ray_origin = Vector4(0.0, 0.0, 0.0, 1.0)

        if self.sensor_dim[3] > 0:
            ray_origin[0] = self.sensor_dim[2] * rand() - self.sensor_dim[2] / 2.0
            ray_origin[1] = self.sensor_dim[2] * rand() - self.sensor_dim[2] / 2.0

        ray_dir_world_space = Vector4(dot(ray_dir, self.iview_c1),
                                      dot(ray_dir, self.iview_c2),
                                      dot(ray_dir, self.iview_c3),
                                      dot(ray_dir, self.iview_c4))
        ray_origin_world_space = Vector4(dot(ray_origin, self.iview_c1),
                                         dot(ray_origin, self.iview_c2),
                                         dot(ray_origin, self.iview_c3),
                                         dot(ray_origin, self.iview_c4))

        final_ray = ray_dir_world_space - ray_origin_world_space
        final_ray = normalize(final_ray)
        ray_origin_world_space_vec3 = Vector(ray_origin_world_space[0],
                                             ray_origin_world_space[1],
                                             ray_origin_world_space[2])
        final_ray_vec3 = Vector(final_ray[0], final_ray[1], final_ray[2])
        return ray_origin_world_space_vec3, final_ray_vec3
