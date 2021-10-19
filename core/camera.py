from pyrr.matrix44 import create_look_at
from random import random
from .ray import Ray
from mathematics.vec3 import normalize_vector, to_homogeneous_vector
from .camera_taichi import Camera as CameraTaichi
from mathematics.vec3_taichi import *

from math import tan, radians
import numpy as np
import sys


class Camera:
    def __init__(self, position, looking_at, up, resolution, fov=90, aperture=0, focal_dist=1.0):
        self.position = np.array(position)
        self.looking_at = np.array(looking_at)
        self.up = np.array(up)
        self.view = create_look_at(self.position, self.looking_at, self.up)
        self.iview = np.linalg.inv(self.view)
        self.resolution = resolution

        self.aperture = aperture
        self.focal_dist = focal_dist
        self.fov = fov
        self.aspect_ratio = self.resolution[0]/self.resolution[1]*1.0

    def convert_to_taichi_camera(self):
        vfrom = Point(self.position[0], self.position[1], self.position[2])
        at = Point(self.looking_at[0], self.looking_at[1], self.looking_at[2])
        up = Point(self.up[0], self.up[1], self.up[2])
        focus_dist = self.focal_dist
        aperture = self.aperture
        aspect_ratio = float(self.resolution[0])/self.resolution[1]
        cam = CameraTaichi(vfrom, at, up, self.fov, aspect_ratio, aperture,
                           focus_dist, self.iview.T)
        return cam

    def get_resolution(self):
        return self.resolution

    def generate_ray(self, screen_coordinates):
        """
        generate a ray for a screen coordinate
        :param screen_coordinates:
        :return:
        """

        sensor_height = tan(radians(self.fov) / 2) * self.focal_dist

        sensor_width = sensor_height * self.aspect_ratio

        cam_space_coord = screen_coordinates - 0.5
        ray_dir = np.array([cam_space_coord[0] * sensor_width / 0.5,
                            cam_space_coord[1] * sensor_height / 0.5,
                            -self.focal_dist])

        ray_origin = np.array([0.0, 0.0, 0.0])

        if self.aperture > 0:
            ray_origin[0] = self.aperture * random() - self.aperture / 2.0
            ray_origin[1] = self.aperture * random() - self.aperture / 2.0

        ray_dir_world_space = to_homogeneous_vector(ray_dir) @ self.iview
        ray_origin_world_space = to_homogeneous_vector(ray_origin) @ self.iview

        # ray_dir_world_space = self.view @ to_homogeneous_vector(ray_dir)
        # ray_origin_world_space = self.view @ to_homogeneous_vector(ray_origin)

        final_ray = ray_dir_world_space - ray_origin_world_space
        final_ray = normalize_vector(final_ray)
        a_ray = Ray(ray_origin_world_space[:3], final_ray[:3], 8)
        return a_ray
