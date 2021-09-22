from pyrr.matrix44 import create_look_at
from random import random
from .ray import Ray
from mathematics.vec3 import normalize_vector
from mathematics.constants import MAX_F
from math import tan, radians
import numpy as np


class Camera:
    def __init__(self, position, looking_at, up, resolution, fov=90, aperture=0, focal_dist=1.0):
        self.position = position
        self.looking_at = looking_at
        self.up = up
        self.iview = create_look_at(self.position, self.looking_at, self.up)
        self.resolution = resolution

        self.aperture = aperture
        self.focal_dist = focal_dist
        self.fov = fov
        self.aspect_ratio = self.resolution[0]/self.resolution[1]*1.0

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

        ray_dir_world_space = self.iview @ ray_dir
        ray_origin_world_space = self.iview @ ray_origin

        final_ray = ray_dir_world_space - ray_origin_world_space
        final_ray = normalize_vector(final_ray)
        return Ray(ray_origin_world_space, final_ray, 8, np.array([0.0, MAX_F]))
