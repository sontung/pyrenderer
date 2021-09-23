import numpy as np
from .constants import GAMMA2_3


class BBox:
    def __init__(self, min_coord, max_coord):
        self.min_coord = min_coord
        self.max_coord = max_coord

    def from_vertices(self, vertices):
        self.min_coord = np.min(vertices, axis=0)
        self.max_coord = np.max(vertices, axis=0)

    def enclose(self, bbox):
        self.min_coord = np.minimum(self.min_coord, bbox.min_coord)
        self.max_coord = np.maximum(self.max_coord, bbox.max_coord)

    def surface_area(self):
        extent = self.max_coord - self.min_coord
        return 2.0 * (extent[0] * extent[2] + extent[0] * extent[1] + extent[1] * extent[2])

    def hit(self, ray):
        res = {"origin": ray.position, "hit": False, "t": 0.0,
               "position": np.array([0.0, 0.0, 0.0])}

        t0 = ray.bounds[0], t1 = ray.bounds[1]
        for i in range(3):
            inv_ray_dir = ray.inv_direction[i]
            t_near = (self.min_coord[i] - ray.position[i]) * inv_ray_dir
            t_far = (self.max_coord[i] - ray.position[i]) * inv_ray_dir
            if t_near > t_far:
                t_near, t_far = t_far, t_near

            t_far *= 1 + 2 * GAMMA2_3

            if t_near > t0:
                t0 = t_near
            if t_far < t1:
                t1 = t_far
            if t0 > t1:
                return res

        res.hit = True
        res.distance = t0
        return res
