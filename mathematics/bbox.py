import numpy as np
from .constants import GAMMA2_3, EPS, MAX_F


class BBox:
    def __init__(self, min_coord=None, max_coord=None):
        if min_coord is None:
            self.min_coord = np.array([MAX_F, MAX_F, MAX_F])
            self.max_coord = self.min_coord*-1
        else:
            self.min_coord = min_coord
            self.max_coord = max_coord
        self.empty = False

    def from_vertices(self, vertices):
        self.min_coord = np.min(vertices, axis=0)
        self.max_coord = np.max(vertices, axis=0)
        self.update_empty()

    def __str__(self):
        return f"bbox: max={self.max_coord} min={self.min_coord}"

    def copy(self, bbox):
        self.min_coord = bbox.min_coord
        self.max_coord = bbox.max_coord
        self.update_empty()

    def update_empty(self):
        self.empty = np.any(np.abs(self.min_coord - self.max_coord) <= EPS)

    def center(self):
        return (self.min_coord+self.max_coord)/2.0

    def is_empty(self):
        return self.empty

    def enclose(self, bbox):
        self.min_coord = np.minimum(self.min_coord, bbox.min_coord)
        self.max_coord = np.maximum(self.max_coord, bbox.max_coord)
        self.update_empty()

    def enclose_point(self, point):
        self.min_coord = np.minimum(self.min_coord, point)
        self.max_coord = np.maximum(self.max_coord, point)
        self.update_empty()

    def surface_area(self):
        extent = self.max_coord - self.min_coord
        return 2.0 * (extent[0] * extent[2] + extent[0] * extent[1] + extent[1] * extent[2])

    def hit(self, ray):
        res = {"origin": ray.position, "hit": False, "t": MAX_F,
               "position": np.array([0.0, 0.0, 0.0])}
        t0 = ray.bounds[0]
        t1 = ray.bounds[1]
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

        res["hit"] = True
        res["t"] = t0
        return res
