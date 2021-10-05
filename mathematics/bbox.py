import numpy as np
from numba import njit
from .constants import GAMMA2_3, EPS, MAX_F


@njit("(f8, f8, f8[:], f8[:], f8[:], f8[:], f8[:])")
def compute(t0, t1, position, inv_direction, min_coord, max_coord, res_holder):
    for i in range(3):
        inv_ray_dir = inv_direction[i]
        t_near = (min_coord[i] - position[i]) * inv_ray_dir
        t_far = (max_coord[i] - position[i]) * inv_ray_dir
        if t_near > t_far:
            t_near, t_far = t_far, t_near

        t_far *= 1 + 2 * GAMMA2_3

        if t_near > t0:
            t0 = t_near
        if t_far < t1:
            t1 = t_far
        if t0 > t1:
            res_holder[0] = -1.0
            return

    res_holder[0] = 1.0
    res_holder[1] = t0


class BBox:
    def __init__(self, min_coord=None, max_coord=None):
        if min_coord is None:
            self.min_coord = np.array([MAX_F, MAX_F, MAX_F])
            self.max_coord = self.min_coord*-1
        else:
            self.min_coord = min_coord
            self.max_coord = max_coord
        self.empty = False
        self.res_holder = np.zeros((2,), np.float64)

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

    # @profile
    def hit(self, ray):
        compute(ray.bounds[0], ray.bounds[1], ray.position, ray.inv_direction,
                self.min_coord, self.max_coord, self.res_holder)
        if self.res_holder[0] > 0.0:
            res = {"hit": True, "t": self.res_holder[1]}
            return res
        else:
            return {"hit": False, "t": 0.0}

